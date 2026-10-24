"""CPU config and actual-loop crossed-context check, explicitly scripted output."""
import asyncio,json,os,runpy
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoProcessor
from omegaconf import OmegaConf
from vagen.agent_loop.fixed_library_agent import FixedLibraryAgent
from verl.utils.config import validate_config
ROOT=Path(os.environ['TASK_ROOT']);NAME='fixed_library_ppo_v1';OUT=ROOT/'artifacts/component_tests'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());DATA=json.loads((OUT/'data_manifest.json').read_text())
for arm in ['A','B']:
 os.environ['TASK_PHASE']=arm
 namespace=runpy.run_path(str(ROOT/'src/scripts/run_fixed_library_ppo.py'),run_name='config_only')
 cfg=namespace['make_config']();validate_config(cfg,use_reference_policy=False,use_critic=True)
 assert cfg.trainer.total_training_steps==4 and cfg.trainer.test_freq==2 and cfg.trainer.save_freq==-1
 assert cfg.algorithm.gamma==cfg.algorithm.lam==1 and cfg.skill_system.reward.guided_weight==0
 (OUT/f'validated_config_{arm}.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
async def main():
 folder=OUT/'cpu_fixture';folder.mkdir(exist_ok=False)
 os.environ.update(PILOT_OUT=str(folder),PILOT_DATA_MANIFEST=str(OUT/'data_manifest.json'),MECHANISM_AUDIT_DIR=str(folder/'audit'),TASK_PHASE='A',FIXED_EMPTY_LIBRARY=str(folder/'empty'),FIXED_CARD_LIBRARY=str(ROOT/P['source_library']))
 proc=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
 skill=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True);skill['library']={'path':str(folder/'empty'),'max_size':32};skill['reward']['guided_weight']=0
 cfg=OmegaConf.create({'actor_rollout_ref':{'rollout':{'multi_turn':{},'prompt_length':8192,'response_length':4096}},'data':{'apply_chat_template_kwargs':{}},'env_registry':{'FrozenLake':'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},'skill_system':skill})
 class Shared(FixedLibraryAgent):
  _class_initialized=False
  _skill_initialized=False
 captures=[]
 class Scripted:
  def __init__(self,context):self.context=context
  async def generate(self,**kw):
   assert len(kw.get('image_data') or [])==(1 if self.context=='empty' else 2)
   captures.append(self.context)
   return SimpleNamespace(token_ids=proc.tokenizer.encode('<think>Explicit scripted CPU fixture, not a model response.</think><answer>Up, Up, Up</answer>',add_special_tokens=False),log_probs=None)
 async def run(context):
  agent=Shared(SimpleNamespace(config=cfg),Scripted(context),proc.tokenizer,proc)
  target=next(x for x in DATA['splits']['dev'] if x['group']=='conditional')
  result=await agent.run({'temperature':0.,'top_p':1.,'max_new_tokens':512},env_name='FrozenLake',config=P['environment'],seed=target['seed'],max_turns=5,response_length_per_turn=512,pilot_phase='dev',pilot_policy_version=0,fixed_context=context,validate=True)
  assert result.extra_fields['fixed_context']==context
  assert result.extra_fields['retrieved_skill_id']==(None if context=='empty' else 'skill_4')
  assert not result.extra_fields['distillation_audit']['model_calls']
  return context
 contexts=await asyncio.gather(*(run(c) for c in ['cards','empty','empty','cards']))
 assert captures.count('cards')==captures.count('empty')==2
 result={'status':'PASS','config_arms_validated':['A','B'],'real_processor_loop_concurrent_contexts':contexts,'scripted_fixture_generate_calls':4,'model_calls':0,'empty_card_provider_no_leakage':True,'A_can_evaluate_cards':True}
 (OUT/'cpu_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
asyncio.run(main())
