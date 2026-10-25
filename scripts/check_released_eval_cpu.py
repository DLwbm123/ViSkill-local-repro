"""Check released configuration/dataset and the transparent recorder without a model."""
import os,json,runpy,asyncio,tempfile,types,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,ray,pytest
from PIL import Image
from omegaconf import OmegaConf
from verl.utils.config import validate_config
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import _determine_fsdp_megatron_base_class
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);OUT=ROOT/'artifacts/reproduction/released_frozenlake_eval_v1';OUT.mkdir(parents=True,exist_ok=True)
cfg,original=runpy.run_path(str(SRC/'scripts/release_eval_config.py'),run_name='config')['config']()
validate_config(cfg,use_reference_policy=False,use_critic=False)
from vagen.gym_agent_dataset import AgenticDataset
from vagen.skills import get_skill_provider
from vagen.released_eval_observer import ObserverActor,RecordedManager
import vagen,verl
assert Path(vagen.__file__).is_relative_to(SRC) and Path(verl.__file__).is_relative_to(SRC)
train=AgenticDataset(cfg.data.train_files,cfg.data);val=AgenticDataset(cfg.data.val_files,cfg.data)
assert len(train)==10000 and len(val)==256 and {r['seed'] for r in train.items}==set(range(1,10001)) and {r['seed'] for r in val.items}==set(range(10001,10257))
assert cfg.actor_rollout_ref.rollout.agent.num_workers==8 and cfg.data.val_batch_size is None
assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature==0 and not cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
assert _determine_fsdp_megatron_base_class([ObserverActor.__mro__]) is Worker
skill_cfg=OmegaConf.to_container(cfg.skill_system,resolve=True);skill_cfg['library']['path']=str(ROOT/'models/r0/skill_library');provider=get_skill_provider(skill_cfg);provider.init_env_type('frozenlake');assert len(provider.library)>0
assert not hasattr(provider,'contract_guard') and not hasattr(provider,'distill_representation')
async def recorder_check(folder):
    for name in ['requests','request_images']:(folder/name).mkdir()
    output=types.SimpleNamespace(token_ids=[1,2]);image=Image.new('RGB',(2,2),'red');captured=[]
    class Native:
        async def generate(self,**kwargs):captured.append(kwargs);return output
    agent=types.SimpleNamespace(observed_out=folder,observed_seed=10001,observed_kind='rollout',tokenizer=types.SimpleNamespace(decode=lambda *a,**k:'fixture'))
    kwargs={'request_id':'fixture','prompt_ids':[3,4],'sampling_params':{'temperature':0.},'image_data':[image]}
    result=await RecordedManager(agent,Native()).generate(**kwargs)
    assert result is output and captured==[kwargs] and json.loads((folder/'requests/0001.json').read_text())['output_ids']==[1,2]
    (folder/'dispatch_count.txt').write_text('1536')
    try:await RecordedManager(agent,Native()).generate(**kwargs)
    except AssertionError:pass
    else:raise AssertionError('Dispatch cap must stop before native call')
    assert len(captured)==1
with tempfile.TemporaryDirectory(dir=OUT) as tmp:asyncio.run(recorder_check(Path(tmp)))
lifecycle=runpy.run_path(str(SRC/'tests/test_frozenlake_lifecycle.py'),run_name='fixture')['test_close_preserves_other_live_rgb_environment'];patch=pytest.MonkeyPatch()
try:lifecycle(patch)
finally:patch.undo()
assert not ray.is_initialized()
result={'status':'CPU_READY','completed_unix':time.time(),'official_train_seed_count':10000,'official_val_seed_count':256,'official_full_val_seed_set':True,'original_algorithm_extensions_absent':True,'transparent_recorder_return_identity_and_pre_dispatch_cap_pass':True,'actor_MRO_correct':True,'released_library_entries':len(provider.library),'loaded_module_paths_use_clean_tree':True,'CPU_RGB_fixture_resets':8,'CPU_RGB_fixture_env_step_calls':1,'CPU_RGB_fixture_primitive_actions':2,'new_model_calls':0,'new_optimizer_steps':0,'scope':'Tiny recorder mock is not a model call. RGB lifecycle fixture is separate from official256 evaluation and its costs.'}
(OUT/'cpu_check.json').write_text(json.dumps(result,indent=2));(OUT/'data_manifest.json').write_text(json.dumps({'train':train.items,'validation':val.items,'scope':'Exact released dataset expansion; no model-based selection, no historical/structural filtering.'},default=lambda value:value.tolist() if hasattr(value,'tolist') else str(value)))
print(json.dumps(result))
