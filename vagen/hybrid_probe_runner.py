"""Initialize the actual hybrid trainer, then evaluate once without any update/save."""
import json
import os
from pathlib import Path
from vagen.main_ppo_skill import TaskRunner
from vagen.ray_trainer_skill import RayPPOTrainerSkill


class ProbeTrainer(RayPPOTrainerSkill):
    def fit(self):
        assert not self.config.skill_system.enable
        assert self.config.trainer.resume_mode == 'disable'
        self.global_steps = 0
        self._load_checkpoint()
        metrics = self._validate()
        assert metrics
        out = Path(os.environ['TASK_ROOT']) / 'artifacts/component_tests/hybrid_request_probe_v1'
        (out / 'hybrid_metrics.json').write_text(json.dumps(metrics, indent=2, default=lambda value: value.item()))
        records = list((out / 'hybrid').glob('*/result.json'))
        assert len(records) == 16
        self._flush_image_dumps()


class ProbeTaskRunner(TaskRunner):
    def run(self, config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer = ProbeTrainer
        return super().run(config)
