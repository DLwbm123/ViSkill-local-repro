"""CPU checks for phase leakage and reward/terminal-token boundary failures."""
from types import SimpleNamespace
import pytest
import torch

from vagen.agent_loop.bootstrap_agent import allow_distillation, check_terminal_tokens, BootstrapGymAgentLoop
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop
from vagen.bootstrap_pilot import check_reward_mask, _PilotTrainerMixin


def test_actual_phase_gate_keeps_dev_out_of_distillation(monkeypatch):
    called = []
    monkeypatch.setattr(GymAgentLoop, '_should_self_distill', lambda self, data: called.append(data) or True)
    agent = object.__new__(BootstrapGymAgentLoop)
    sample = SimpleNamespace()
    for phase, expected in [('dev', False), ('train', True), ('probe', True)]:
        agent.pilot_phase = phase
        assert agent._should_self_distill(sample) is expected
    assert called == [sample, sample]
    assert sample.distillation_audit['reason'] == 'pilot_dev_phase_disabled'
    with pytest.raises(ValueError):
        allow_distillation('unknown')


def test_terminal_output_and_observation_masks():
    audit = {'spans': [{'kind': 'generated', 'start': 0, 'end': 2},
                       {'kind': 'observation', 'start': 2, 'end': 4},
                       {'kind': 'generated', 'start': 4, 'end': 6}]}
    check_terminal_tokens(audit, list(range(6)), [1, 1, 0, 0, 1, 1])
    with pytest.raises(AssertionError, match='truncated'):
        check_terminal_tokens(audit, list(range(5)), [1, 1, 0, 0, 1])
    with pytest.raises(AssertionError):
        check_terminal_tokens(audit, list(range(6)), [1, 1, 1, 0, 1, 1])


def test_reward_must_land_on_generated_token():
    b = SimpleNamespace(batch={'rm_scores': torch.tensor([[0., 0., 1.]]),
                               'response_mask': torch.tensor([[1, 0, 1]])})
    check_reward_mask(b)
    b.batch['rm_scores'] = torch.tensor([[0., 1., 0.]])
    with pytest.raises(AssertionError, match='observation'):
        check_reward_mask(b)


def test_eval_rejects_library_mutation():
    class FakeBase:
        def _validate(self):
            if self.mutate:
                self.library = {'utility': .6}
            return {'score': 1.0}

    class FakeTrainer(_PilotTrainerMixin, FakeBase):
        def _pilot_library(self):
            return self.library

    trainer = FakeTrainer()
    trainer.library = {'utility': .5}
    trainer.mutate = False
    assert trainer._validate() == {'score': 1.0}
    trainer.mutate = True
    with pytest.raises(AssertionError, match='modified library'):
        trainer._validate()


def test_real_loop_phase_control_with_cpu_scripted_fixture(tmp_path, monkeypatch):
    import asyncio
    import json
    import os
    from pathlib import Path
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from vagen.envs.frozenlake.frozenlake_env import FrozenLake
    from vagen.skills.mechanism_audit import jsonable

    processor = AutoProcessor.from_pretrained(Path(os.environ['TASK_ROOT']) / 'models/b0', local_files_only=True)
    env_config = {'render_mode': 'vision', 'size': 4, 'p': .8, 'is_slippery': False, 'slip_prob': 0.,
                  'max_actions_per_step': 5, 'prompt_format': 'free_think', 'accurate_dynamics_prompt': True,
                  'desc': ['SFGF', 'FFFF', 'FFFF', 'FFFF']}
    strategy = json.dumps({'when': {'goal_direction': 'right', 'goal_manhattan_distance': 2,
                                   'requires_clear_route': True}, 'what': [{'action': 'right', 'count': 2}]})

    async def run_phase(phase):
        class Isolated(BootstrapGymAgentLoop):
            _class_initialized = False
            _skill_initialized = False

        folder = tmp_path / phase
        folder.mkdir()
        monkeypatch.setenv('PILOT_OUT', str(folder))
        monkeypatch.setenv('MECHANISM_AUDIT_DIR', str(folder / 'audit'))
        monkeypatch.setenv('TASK_PHASE', 'A')
        env = FrozenLake(env_config)
        await env.reset(17)
        state = jsonable(env.get_state_for_skill())
        await env.close()
        manifest = folder / 'manifest.json'
        manifest.write_text(json.dumps({'splits': {phase: [{'seed': 17, 'state': state}]}}))
        monkeypatch.setenv('PILOT_DATA_MANIFEST', str(manifest))
        config = OmegaConf.create({'actor_rollout_ref': {'rollout': {'multi_turn': {},
            'prompt_length': 8192, 'response_length': 4096}}, 'data': {'apply_chat_template_kwargs': {}},
            'env_registry': {'FrozenLake': 'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},
            'skill_system': {'enable': True, 'cold_start': {'enable': False},
                'library': {'path': str(folder / 'empty_library')},
                'distill': {'mode': 'self', 'representation': 'directional_contract_v1',
                    'allow_assistant_envelope': True, 'quality_ratio': 1.},
                'reward': {'mode': 'state_progress_bounded', 'guided_weight': 0.},
                'retrieval': {'contract_guard': True}}})

        class ScriptedManager:
            async def generate(self, **kwargs):
                text = strategy if kwargs['sampling_params']['max_new_tokens'] <= 256 else '<think>fixture only</think><answer>Right, Right</answer>'
                return SimpleNamespace(token_ids=processor.tokenizer.encode(text, add_special_tokens=False), log_probs=None)

        agent = Isolated(SimpleNamespace(config=config), ScriptedManager(), processor.tokenizer, processor)
        result = await agent.run({'temperature': 0. if phase != 'train' else 1., 'top_p': 1., 'max_new_tokens': 512},
            env_name='FrozenLake', config=env_config, seed=17, validate=phase != 'train',
            max_turns=5, response_length_per_turn=512, pilot_phase=phase, pilot_policy_version=0)
        assert result.extra_fields['traj_success']
        count = int((folder / 'model_calls.txt').read_text())
        assert count == (1 if phase == 'dev' else 2)
        assert bool(result.extra_fields['distilled_strategy']) is (phase != 'dev')
        records = [json.loads(p.read_text()) for p in (folder / 'requests').glob('*.json')]
        assert all(r['sampling']['temperature'] == (0. if phase == 'dev' else 1.)
                   for r in records if r['kind'] == 'rollout')

    for phase in ['train', 'dev', 'probe']:
        asyncio.run(run_phase(phase))
