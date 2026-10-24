"""Experiment-only phase control and raw evidence for the paired PPO pilot."""
import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4

from .gym_agent_loop_skill import GymAgentLoop
from vagen.skills.mechanism_audit import jsonable, library_snapshot


def allow_distillation(phase):
    if phase not in {"train", "dev", "probe"}:
        raise ValueError(f"unknown pilot phase: {phase}")
    return phase != "dev"


def check_terminal_tokens(audit, response_ids, response_mask):
    generated = [s for s in audit["spans"] if s["kind"] == "generated"]
    assert generated, "no generated terminal output"
    assert generated[-1]["end"] <= len(response_ids), "terminal output truncated"
    assert len(response_ids) == len(response_mask)
    if response_mask:
        assert response_mask[-1] == 1, "episode ends in an observation token"
    for span in audit["spans"]:
        expected = int(span["kind"] == "generated")
        assert all(x == expected for x in response_mask[span["start"]:span["end"]])


class _RecordedManager:
    def __init__(self, agent, original):
        self.agent, self.original = agent, original

    async def generate(self, **kwargs):
        agent = self.agent
        counter = agent.pilot_out / "model_calls.txt"
        with counter.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            call = int(stream.read() or "0") + 1
            assert call <= 1600, "pilot model-call limit"
            stream.seek(0)
            stream.truncate()
            stream.write(str(call))
            stream.flush()
        output = await self.original.generate(**kwargs)
        record = {"call": call, "seed": agent.pilot_seed,
                  "phase": agent.pilot_phase, "policy_version": agent.pilot_version,
                  "kind": "distill" if agent.pilot_distilling else "rollout",
                  "request_id": kwargs["request_id"],
                  "sampling": kwargs["sampling_params"],
                  "prompt_ids": kwargs["prompt_ids"],
                  "image_count": len(kwargs.get("image_data") or []),
                  "output_ids": output.token_ids,
                  "raw": agent.tokenizer.decode(output.token_ids, skip_special_tokens=True)}
        (agent.pilot_out / "requests" / f"{call:04d}.json").write_text(
            json.dumps(jsonable(record)))
        return output


class BootstrapGymAgentLoop(GymAgentLoop):
    async def run(self, sampling_params, **kwargs):
        self.pilot_phase = kwargs["pilot_phase"]
        allow_distillation(self.pilot_phase)
        self.pilot_seed = int(kwargs["seed"])
        self.pilot_version = int(kwargs["pilot_policy_version"])
        self.pilot_out = Path(os.environ["PILOT_OUT"])
        self.pilot_distilling = False
        self.pilot_injected_image = None
        if self.pilot_phase == "probe":
            # Remote rollout workers retain their original greedy Dev config.
            sampling_params = dict(sampling_params, temperature=1.0, top_p=1.0)
        assert sampling_params["temperature"] == (0.0 if self.pilot_phase == "dev" else 1.0)
        (self.pilot_out / "requests").mkdir(parents=True, exist_ok=True)
        original = self.server_manager
        self.server_manager = _RecordedManager(self, original)
        try:
            output = await super().run(sampling_params, **kwargs)
        finally:
            self.server_manager = original
        row = output.extra_fields
        audit = row["mechanism_audit"]
        check_terminal_tokens(audit, output.response_ids, output.response_mask)
        assert audit["distillation_separate_window"] and abs(audit["skill_bonus"]) < 1e-8
        assert self.skill_provider.guided_weight == 0
        if self.pilot_phase == "dev":
            assert not (row.get("distillation_audit") or {}).get("model_calls", 0)
        if os.environ["TASK_PHASE"] == "A":
            assert row["retrieved_skill_id"] is None
            assert not self.skill_provider.library._skills
        expected = json.loads(Path(os.environ["PILOT_DATA_MANIFEST"]).read_text())
        target = next(x for x in expected["splits"][self.pilot_phase] if x["seed"] == self.pilot_seed)
        assert jsonable(row["initial_env_state"]) == target["state"], "actual map differs from frozen input"
        identity = uuid4().hex
        folder = self.pilot_out / "episodes" / f"{self.pilot_phase}_{self.pilot_version}_{self.pilot_seed}_{identity}"
        folder.mkdir(parents=True)
        frames = row["trajectory_frames"]
        if frames:
            frames[0].save(folder / "initial.png")
            frames[-1].save(folder / "terminal.png")
        if row.get("validation_skill_image") is not None:
            row["validation_skill_image"].save(folder / "injected_skill.png")
        elif self.pilot_injected_image is not None:
            self.pilot_injected_image.save(folder / "injected_skill.png")
        serial = {k: v for k, v in row.items() if k not in {
            "trajectory_frames", "validation_skill_image", "validation_initial_state_image"}}
        serial.update(phase=self.pilot_phase, policy_version=self.pilot_version,
                      distill_policy_version=self.pilot_version,
                      reward_score=output.reward_score, frame_count=len(frames),
                      bfs=self.skill_provider.bfs_distance(row["initial_env_state"]),
                      library=library_snapshot(self.skill_provider))
        skill = audit.get("retrieved_skill")
        reference = skill.get("positions", []) if skill else []
        positions = row["trajectory_positions"]
        progress = 0
        if reference and positions:
            relative = lambda ps: [(int(r)-int(ps[0][0]), int(c)-int(ps[0][1])) for r, c in ps]
            visited = set(relative(positions))
            progress = max(i for i, state in enumerate(relative(reference)) if state in visited)
        from vagen.skills.reward import compute_state_progress_reward
        shadow = .8 * compute_state_progress_reward(reference, positions) if reference and not row["traj_success"] else 0.0
        length = len(skill.get("trajectory", [])) if skill else 0
        serial["reward_support"] = {"reference_length": length, "furthest_progress": progress,
            "executed_actions": len(row["trajectory_actions"]), "shadow_bonus_lambda_0_8": shadow,
            "applied_skill_bonus": audit["skill_bonus"],
            "reason": "success_no_bonus" if row["traj_success"] else "no_retrieval" if not skill else
                "single_step_reference" if length == 1 else "no_progress" if progress == 0 else
                "action_cost_cancels_progress" if shadow == 0 else "positive_shadow_only"}
        board = row["initial_env_state"]["desc"]
        last = positions[-1] if positions else row["initial_env_state"]["player_pos"]
        serial["terminal_reason"] = ("goal" if row["traj_success"] else
            "hole" if board[int(last[0])][int(last[1])] == "H" else
            "turn_limit" if output.num_turns >= 5 else "token_or_other_limit")
        (folder / "episode.json").write_text(json.dumps(jsonable(serial)))
        row["pilot_phase"] = self.pilot_phase
        row["pilot_policy_version"] = self.pilot_version
        row["pilot_episode_path"] = str(folder / "episode.json")
        return output

    def _inject_skill(self, agent_data, skill, **kwargs):
        injected = super()._inject_skill(agent_data, skill, **kwargs)
        if injected and agent_data.image_data:
            self.pilot_injected_image = agent_data.image_data[-1].copy()
        return injected

    def _should_self_distill(self, agent_data):
        if not allow_distillation(self.pilot_phase):
            agent_data.distillation_audit = {"model_calls": 0, "reason": "pilot_dev_phase_disabled"}
            return False
        return super()._should_self_distill(agent_data)

    async def _self_distill(self, *args, **kwargs):
        self.pilot_distilling = True
        try:
            return await super()._self_distill(*args, **kwargs)
        finally:
            self.pilot_distilling = False
