"""Scoped trainer hooks; the existing PPO algorithms and callbacks are reused."""
import json
import os
import time
from pathlib import Path

import numpy as np
from torchdata.stateful_dataloader import StatefulDataLoader

from vagen.gym_agent_dataset import AgenticDataset
from vagen.ray_trainer import RayPPOTrainer
from vagen.ray_trainer_skill import RayPPOTrainerSkill
from vagen.skills.mechanism_audit import jsonable


class PilotDataset(AgenticDataset):
    def __init__(self, data_files, config, **kwargs):
        super().__init__(data_files=data_files, config=config, **kwargs)
        phase = Path(str(data_files)).stem.rsplit("_", 1)[-1]
        assert phase in {"train", "dev", "probe"}
        for item in self.items:
            item["pilot_phase"] = phase


def check_reward_mask(batch):
    rewards, mask = batch.batch["rm_scores"], batch.batch["response_mask"]
    assert bool(((rewards == 0) | (mask == 1)).all()), "reward assigned to observation/padding"


class _AuditedRollout:
    def __init__(self, trainer, original):
        self.trainer, self.original = trainer, original
        self.sequence = 0

    def __getattr__(self, key):
        return getattr(self.original, key)

    def generate_sequences(self, batch):
        self.sequence += 1
        started = time.time()
        output = self.original.generate_sequences(batch)
        check_reward_mask(output)
        data = output.non_tensor_batch
        rows = []
        for i in range(len(output)):
            audit = data["mechanism_audit"][i]
            assert audit["distillation_separate_window"]
            mmi = data.get("multi_modal_inputs", [None] * len(output))[i] or {}
            image_grid = mmi.get("image_grid_thw")
            image_count = int(image_grid.shape[0]) if image_grid is not None else 0
            if data["retrieved_skill_id"][i]:
                assert image_count >= 2, "injected card missing from processed multimodal input"
            rows.append({"seed": data["seed"][i], "phase": data["pilot_phase"][i],
                         "policy_version": data["pilot_policy_version"][i],
                         "retrieved_skill_id": data["retrieved_skill_id"][i],
                         "episode": data["pilot_episode_path"][i],
                         "processed_image_count": image_count,
                         "generated_response_tokens": int(output.batch["response_mask"][i].sum()),
                         "reward": float(output.batch["rm_scores"][i].sum()),
                         "reward_token_indices": output.batch["rm_scores"][i].nonzero().flatten().tolist()})
        self.trainer._pilot_record("dispatches", str(self.sequence), {
            "start": started, "end": time.time(), "rows": rows,
            "actual_manager_generate_returned": True, "reward_mask_checked": True})
        return output


class _PilotTrainerMixin:
    def _pilot_record(self, kind, identity, value):
        p = Path(os.environ["PILOT_OUT"]) / kind
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{identity}.json").write_text(json.dumps(jsonable(value), indent=2))

    def init_workers(self):
        assert len(self.train_dataloader) == 4 and len(self.train_dataset) == 128
        assert self.config.trainer.save_freq == -1
        assert not self._hf_upload_manager.should_upload(4)
        super().init_workers()
        self.async_rollout_manager = _AuditedRollout(self, self.async_rollout_manager)

    def _save_checkpoint(self):
        raise AssertionError("checkpoint save forbidden in this diagnostic")

    def _get_gen_batch(self, batch):
        result = super()._get_gen_batch(batch)
        phases = set(result.non_tensor_batch["pilot_phase"])
        assert len(phases) == 1
        phase = next(iter(phases))
        version = min(4, self.global_steps) if phase != "train" else self.global_steps - 1
        result.non_tensor_batch["pilot_policy_version"] = np.full(len(result), version, dtype=object)
        return result

    def _validate(self):
        before = self._pilot_library()
        metrics = super()._validate()
        assert before == self._pilot_library(), "evaluation modified library or utility"
        return metrics

    def _pilot_library(self):
        path = Path(self.config.skill_system.library.path) / "skill_index.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        check_reward_mask(batch)
        self._pilot_record("updated_batches", str(self.global_steps), {
            "rollout_policy_version": self.global_steps - 1,
            "optimizer_update_completed_version": self.global_steps,
            "seeds": list(batch.non_tensor_batch["seed"]),
            "skill_ids": list(batch.non_tensor_batch["retrieved_skill_id"]),
            "episode_paths": list(batch.non_tensor_batch["pilot_episode_path"]),
            "response_mask_checked": True,
            "library_after_callback": self._pilot_library()})
        # Preserve summaries without duplicating all frames in standard rollout dumps.
        for key in ("seed", "pilot_phase", "pilot_policy_version", "pilot_episode_path", "retrieved_skill_id"):
            reward_extra_infos_dict[key] = jsonable(list(batch.non_tensor_batch[key]))
        super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

    def fit(self):
        super().fit()
        assert self.global_steps == 5, "four update batches not completed"
        assert len(list((Path(os.environ["PILOT_OUT"]) / "updated_batches").glob("*.json"))) == 4
        from verl.utils.dataset.rl_dataset import collate_fn
        probe = PilotDataset(self.config.bootstrap_pilot.probe_file, self.config.data)
        assert len(probe) == 32
        self.val_dataloader = StatefulDataLoader(probe, batch_size=8, shuffle=False,
            drop_last=False, num_workers=0, collate_fn=collate_fn)
        self.config.actor_rollout_ref.rollout.val_kwargs.temperature = 1.0
        self.config.actor_rollout_ref.rollout.val_kwargs.top_p = 1.0
        self.config.actor_rollout_ref.rollout.val_kwargs.do_sample = True
        self.config.trainer.validation_data_dir = str(Path(os.environ["PILOT_OUT"]) / "probe_validation")
        before = self._pilot_library()
        probe_metrics = self._validate()
        assert self.global_steps == 5 and before == self._pilot_library()
        self._pilot_record("completion", "trainer", {"global_step_after_fit": self.global_steps,
            "training_batches": 4, "probe_completed": True, "probe_metrics": probe_metrics,
            "library": self._pilot_library(), "checkpoint_saved": False,
            "final_model_retained": False, "independent_recovery_verified": False})


class PilotPlainTrainer(_PilotTrainerMixin, RayPPOTrainer):
    pass


class PilotCardTrainer(_PilotTrainerMixin, RayPPOTrainerSkill):
    pass


from vagen.main_ppo_skill import TaskRunner


class PilotTaskRunner(TaskRunner):
    def run(self, config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer = PilotPlainTrainer if config.bootstrap_pilot.arm == "A" else PilotCardTrainer
        return super().run(config)
