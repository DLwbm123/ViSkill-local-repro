"""Concurrent RGB environments must not tear down each other's SDL runtime."""
import asyncio
from vagen.envs.frozenlake.frozenlake_env import FrozenLake


def test_close_preserves_other_live_rgb_environment(monkeypatch):
    from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

    def forbidden_global_close(self):
        raise AssertionError('per-episode close must not call global pygame.quit')

    monkeypatch.setattr(FrozenLakeEnv, 'close', forbidden_global_close)

    async def run():
        cfg = {'render_mode': 'vision', 'desc': ['SFGF', 'FFFF', 'FFFF', 'FFFF'],
               'is_slippery': False, 'slip_prob': 0., 'max_actions_per_step': 5}
        envs = [FrozenLake(cfg) for _ in range(8)]
        await asyncio.wait_for(asyncio.gather(*(e.reset(i) for i, e in enumerate(envs))), timeout=10)
        await asyncio.wait_for(asyncio.gather(*(e.close() for e in envs[:7])), timeout=2)
        obs, reward, done, info = await asyncio.wait_for(
            envs[-1].step('<think>fixture</think><answer>Right, Right</answer>'), timeout=5)
        assert done and info['success']
        assert obs['multi_modal_input']['<image>']
        await envs[-1].close()
        assert all(e.gym_env is None for e in envs)

    asyncio.run(run())
