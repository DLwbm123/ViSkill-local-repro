"""One owned-actor lifecycle repair; never repeat inference, updates or resets."""
import json
import os
from pathlib import Path
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.environ['RAY_USAGE_STATS_ENABLED'] = '0'
import ray
import ray._private.services
import ray._private.state
from ray.core.generated import gcs_pb2

ROOT = Path(os.environ['TASK_ROOT'])
receipt = json.loads((ROOT / 'state/components/bootstrap_card_pair_v1_receipt.json').read_text())
assert not (ROOT / 'state/components/bootstrap_card_pair_v1_EXIT.json').exists()
# The existing isolated cluster is bound to loopback; only this diagnostic
# driver's address normalization is overridden, not cluster networking.
ray._private.services.resolve_ip_for_localhost = lambda host: host
ray.init(address='127.0.0.1:53659', _node_ip_address='127.0.0.1',
         _temp_dir='/data/local/tmp/z0/pa', log_to_driver=False)
actor = next(a for a in ray._private.state.actors().values() if a.get('Pid') == 4168995)
raw = ray._private.state.state.global_state_accessor.get_actor_info(ray.ActorID.from_hex(actor['ActorID']))
info = gcs_pb2.ActorTableData.FromString(raw)
handle = ray.get_actor(info.name, namespace=info.ray_namespace)


def repair_owned_cleanup(actor, expected_token):
    import asyncio
    import importlib
    import os
    import threading
    import time
    from pathlib import Path
    import vagen.envs.frozenlake.frozenlake_env as module

    assert os.getpid() == 4168995 and os.environ['TASK_RUN_TOKEN'] == expected_token
    out = Path(os.environ['PILOT_OUT'])
    assert int((out / 'model_calls.txt').read_text()) == 54
    assert not list((out / 'updated_batches').glob('*.json'))
    assert not list((out / 'update_audit').glob('*.jsonl'))
    matches = []
    for task in list(asyncio.tasks._all_tasks):
        if task.done():
            continue
        coroutine = task.get_coro()
        seen = set()
        close_env = None
        seed = None
        phase = None
        functions = []
        while coroutine is not None and id(coroutine) not in seen:
            seen.add(id(coroutine))
            frame = getattr(coroutine, 'cr_frame', None)
            if frame:
                functions.append(frame.f_code.co_name)
                if frame.f_code.co_filename.endswith('/bootstrap_agent.py') and frame.f_code.co_name == 'run':
                    agent = frame.f_locals['self']
                    seed, phase = agent.pilot_seed, agent.pilot_phase
                if frame.f_code.co_filename.endswith('/frozenlake_env.py') and frame.f_code.co_name == 'close':
                    close_env = frame.f_locals['self']
            coroutine = getattr(coroutine, 'cr_await', None) or getattr(coroutine, 'gi_yieldfrom', None)
        if close_env is not None:
            assert seed == 61010 and phase == 'dev' and functions[-1] == 'to_thread'
            matches.append((task, close_env))
    assert len(matches) == 1, 'only the verified cleanup waiter may be completed'
    task, env = matches[0]
    waiter = task._fut_waiter
    assert isinstance(waiter, asyncio.Future) and not waiter.done()
    old_class = module.FrozenLake
    fixed_class = importlib.reload(module).FrozenLake
    old_class.close = fixed_class.close
    journal = {'time': time.time(), 'actor_pid': os.getpid(), 'seed': 61010,
        'phase': 'dev', 'model_calls_before': 54, 'completed_episodes_before': 15,
        'optimizer_updates_before': 0,
        'repair': 'Release RGB-env references and complete only its observed stuck pygame.quit cleanup waiter.',
        'inference_repeated': False, 'environment_actions_repeated': False,
        'request_outputs_edited': False, 'clock_or_budget_reset': False,
        'remaining_thread': 'Original pygame.quit C call may remain blocked until owned worker shutdown.'}
    journal_path = Path(os.environ['TASK_ROOT']) / 'state/components/bootstrap_close_repair_v1.json'
    journal_path.write_text(json.dumps(journal, indent=2))
    completed = threading.Event()

    def finish_cleanup():
        env.gym_env.window = None
        env.gym_env.clock = None
        env.gym_env = None
        assert not waiter.done()
        waiter.set_result(None)
        completed.set()

    task.get_loop().call_soon_threadsafe(finish_cleanup)
    assert completed.wait(5), 'cleanup completion was not dispatched'
    journal['cleanup_waiter_completed'] = True
    journal_path.write_text(json.dumps(journal, indent=2))
    return journal


try:
    print(json.dumps(ray.get(handle.__ray_call__.remote(repair_owned_cleanup, receipt['run_token']), timeout=15), indent=2))
finally:
    ray.shutdown()
