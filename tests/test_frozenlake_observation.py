import asyncio
import numpy as np
from vagen.envs.frozenlake.frozenlake_env import FrozenLake


def test_true_environment_successful_fixture(trajectory):
    async def run():
        env = FrozenLake({'render_mode':'vision','desc':[''.join(x) for x in trajectory['state']['desc']]})
        try:
            obs,_ = await env.reset(1)
            im=obs['multi_modal_input']['<image>'][0]
            assert im.size==(256,256) and np.asarray(im).std()>10
            for a in trajectory['actions']:
                obs,reward,done,info=await env.step(f'<think>test fixture</think><answer>{a}</answer>')
                assert info['actions']==[a]
            assert done and info['success'] and reward==1.02
        finally: await env.close()
    asyncio.run(run())
