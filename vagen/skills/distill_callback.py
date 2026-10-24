"""
Post-step skill distillation callback.

Distills successful trajectories into skills, updates utility scores,
and manages the skill library.
"""

import logging
import os
from uuid import uuid4
from typing import Optional

logger = logging.getLogger(__name__)


class DistillCallback:

    def __init__(self, skill_cfg: dict):
        from vagen.skills import get_skill_provider
        self.provider = get_skill_provider(dict(skill_cfg))

    def step(self, batch) -> dict:
        rollout_data = self._extract_rollout_data(batch)
        if not rollout_data:
            return {}

        env_names = batch.non_tensor_batch.get("env_name")
        self._env_type = ""
        if env_names is not None and len(env_names) > 0:
            self._env_type = str(env_names[0]).lower()
            self.provider.init_env_type(self._env_type)

        if self.provider.library is None:
            return {}

        from .mechanism_audit import library_snapshot, write_record
        before = library_snapshot(self.provider) if os.getenv("MECHANISM_AUDIT_DIR") else None

        # Update utility for all retrieved skills
        for r in rollout_data:
            skill_id = r.get("retrieved_skill_id")
            if skill_id:
                reward = 1.0 if r.get("traj_success", False) else 0.0
                self.provider.update_utility(skill_id, reward)

        # Distill successful trajectories into new skills
        candidates = self._filter_distill_candidates(rollout_data)
        admitted = 0
        for r in candidates:
            if self.provider.recheck_batch_novelty:
                # Use the live in-memory library: provider.retrieve reloads disk
                # and would discard skills inserted earlier in this same batch.
                nearest = self.provider.library.retrieve(r["initial_env_state"])
                nearest = self.provider.validate_retrieved_skill(nearest, r["initial_env_state"])
                if nearest and nearest["score"] >= self.provider.distill_min_score:
                    r["admission_audit"] = "batch_novelty"
                    continue
            skill_data = self._build_skill_data(r)
            skill_id = self.provider.library.admit(skill_data) if skill_data else None
            if skill_id:
                r["admission_audit"] = "admitted"
                r["admitted_skill_id"] = skill_id
                admitted += 1
            elif r.get("admission_audit") == "candidate":
                r["admission_audit"] = "invalid_skill_data"

        self.provider.save()
        if os.getenv("MECHANISM_AUDIT_DIR"):
            write_record("admissions", uuid4().hex, {"before": before,
                "after": library_snapshot(self.provider), "rows": [
                    {k: v for k, v in row.items() if k not in ("trajectory_frames", "skill_artifact")}
                    for row in rollout_data]})

        return self._collect_metrics(rollout_data, admitted)

    def _filter_distill_candidates(self, rollout_data: list) -> list:
        """Filter for high-quality successes that aren't already covered by existing skills."""
        candidates = []
        seen_seeds = set()

        for r in rollout_data:
            r["admission_audit"] = "not_success"
            if not r.get("traj_success", False):
                continue

            actions = r.get("trajectory_actions", [])
            if self.provider.env_type == "maniskill":
                artifact = r.get("skill_artifact") or {}
                actions = artifact.get("actions", [])
                if not actions:
                    r["admission_audit"] = "no_actions"
                    continue

            # Quality check: trajectory must be near-optimal
            initial_state = r.get("initial_env_state")
            bfs_steps = self.provider.bfs_distance(initial_state) if initial_state else None
            r["bfs_steps"] = bfs_steps
            r["quality_ratio_observed"] = len(actions)/bfs_steps if bfs_steps else None
            rejection = self.provider.distill_rejection(len(actions), bfs_steps, r.get("skill_score"))
            if rejection:
                r["admission_audit"] = rejection
                continue

            # One per seed
            seed = r.get("seed")
            if seed in seen_seeds:
                r["admission_audit"] = "duplicate_seed"
                continue
            seen_seeds.add(seed)
            r["admission_audit"] = "candidate"
            candidates.append(r)

        return candidates

    def _build_skill_data(self, r: dict) -> Optional[dict]:
        env_state = r.get("initial_env_state")
        actions = r.get("trajectory_actions", [])
        artifact = r.get("skill_artifact")
        if self.provider.env_type == "maniskill":
            actions = artifact.get("actions", []) if artifact else []
        if not actions or env_state is None:
            return None

        features = self.provider.extract_features(env_state)
        if not features:
            return None

        frames = r.get("trajectory_frames", [])
        positions = r.get("trajectory_positions", [])
        init_frame = frames[0] if frames else None
        strategy = self._distill_strategy(
            r, init_frame, frames, actions, positions, artifact)
        if self.provider.distill_representation == "directional_contract_v1":
            prior = r.get("distillation_audit") or {}
            if not strategy and prior.get("valid") is False:
                r["admission_audit"] = "invalid_directional_strategy"
                r["admission_error"] = prior.get("reason", "self-distillation rejected")
                return None
            from .directional_contract import require_deterministic_state, validate_directional_observation
            try:
                require_deterministic_state(env_state)
                validate_directional_observation(strategy, r,
                    allow_assistant_envelope=self.provider.allow_assistant_envelope)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                r["admission_audit"] = "invalid_directional_strategy"
                r["admission_error"] = str(exc)
                logger.warning("Rejected invalid directional strategy: %s", exc)
                return None
        from vagen.skills.provider import append_strategy_suffix
        strategy = append_strategy_suffix(self.provider.env_type, strategy)
        trajectory_image = self.provider.render_trajectory(
            frames, actions, positions=positions, strategy=strategy, artifact=artifact)
        skill_data = {
            "features": features,
            "strategy": strategy,
            "image": trajectory_image,
            "seed": r.get("seed"),
            "env_id": env_state.get("env_id"),
        }
        skill_data["trajectory"] = actions
        if self.provider.env_type == "frozenlake" and len(positions) == len(actions)+1:
            skill_data["positions"] = positions
        return skill_data

    def _distill_strategy(self, r, init_frame, frames, actions, positions, artifact) -> str:
        strategy = r.get("distilled_strategy", "")
        if strategy:
            return strategy
        if self.provider.distill_mode == "api" and self.provider.api_distill_enabled:
            trajectory_image = self.provider.render_trajectory(
                frames, actions, positions=positions, artifact=artifact)
            return self.provider.distill_api({
                "trajectory": actions,
                "init_frame": init_frame,
                "trajectory_image": trajectory_image,
                "task_description": self.provider.task_description_for(
                    r.get("initial_env_state")),
                "env_type": self._env_type,
                "env_id": (r.get("initial_env_state") or {}).get("env_id", ""),
            })
        return ""

    def _collect_metrics(self, rollout_data: list, admitted: int) -> dict:
        scores = [r["skill_score"] for r in rollout_data if r.get("skill_score") is not None]
        retrieved = sum(1 for r in rollout_data if r.get("retrieved_skill_id"))
        return {
            "skill/admitted": admitted,
            "skill/library_size": len(self.provider.library),
            "skill/retrieved_count": retrieved,
            "skill/avg_utility": self.provider.library.avg_utility(),
            "skill/avg_score": sum(scores) / len(scores) if scores else 0.0,
        }

    @staticmethod
    def _extract_rollout_data(batch) -> list:
        ntb = batch.non_tensor_batch
        n = len(ntb.get("traj_success", []))
        if n == 0:
            return []

        keys = [
            "traj_success", "retrieved_skill_id", "skill_score",
            "initial_env_state", "seed", "trajectory_actions", "trajectory_frames",
            "trajectory_positions", "skill_artifact", "distilled_strategy", "distillation_audit", "mechanism_audit",
        ]
        rollout_data = []
        for i in range(n):
            entry = {}
            for key in keys:
                arr = ntb.get(key)
                if arr is not None and len(arr) > i:
                    entry[key] = arr[i]
            rollout_data.append(entry)

        return rollout_data
