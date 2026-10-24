"""CPU-only real processor/worker check with explicit scripted responses, not policy evidence."""
import asyncio,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor
from vagen.skills.mechanism_audit import jsonable
from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase, _agent_loop_registry
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop

ROOT=Path(os.environ['TASK_ROOT']);OUT=ROOT/'artifacts/component_tests/mechanism_p0_v1_recheck2'
OUT.mkdir(parents=True,exist_ok=False)
os.environ['MECHANISM_AUDIT_DIR']=str(OUT/'audit')
os.environ['MECHANISM_POLICY_LABEL']='scripted_fixture_no_policy'
processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
skill=OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system
skill.library={'path':str(ROOT/'artifacts/component_tests/directional_native_loop_v1_recovery/library'),'max_size':32}
config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'multi_turn':{},'prompt_length':8192,'response_length':4096}},
 'data':{'apply_chat_template_kwargs':{}},'env_registry':{'FrozenLake':'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},
 'skill_system':skill,'reward_model':{'enable':True,'enable_resource_pool':False}})
_agent_loop_registry['p0_fixture']={'_target_':'vagen.agent_loop.gym_agent_loop_skill.GymAgentLoop'}
class Manager:
 def __init__(self):self.calls=[]
 async def generate(self,request_id,prompt_ids,sampling_params,image_data=None,**kwargs):
  self.calls.append({'ids':list(prompt_ids),'images':list(image_data or [])})
  text='<think>Fixture.</think><answer>None</answer>' if len(self.calls)==1 else '<think>Fixture.</think><answer>Down</answer>'
  return SimpleNamespace(token_ids=processor.tokenizer.encode(text,add_special_tokens=False),log_probs=None)
async def main():
 manager=Manager();handle=SimpleNamespace(config=config,server_manager=manager,tokenizer=processor.tokenizer,processor=processor,reward_router_address=None)
 env=json.loads((ROOT/'src/configs/low_resource/local_eval.json').read_text())['environment']
 result=await AgentLoopWorkerBase._run_agent_loop(handle,{'temperature':0,'max_new_tokens':512},
  {'step':0,'sample_index':0,'rollout_n':0,'validate':False},agent_name='p0_fixture',env_name='FrozenLake',config=env,seed=6,max_turns=2,response_length_per_turn=512)
 audit=result.extra_fields['mechanism_audit'];assert audit['retrieved_skill'] is not None
 assert len(manager.calls)==2
 response=result.response_ids[0];mask=result.response_mask[0]
 for span in audit['spans']:
  end=min(span['end'],len(mask));expected=1 if span['kind']=='generated' else 0
  assert torch.all(mask[span['start']:end]==expected)
 full_loss=torch.cat([torch.zeros(8192,dtype=mask.dtype),mask])
 image_id=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
 assert torch.all(full_loss[result.input_ids[0]==image_id]==0)
 card=manager.calls[0]['images'][0]
 actual=result.multi_modal_data['image'][0]
 assert card is actual
 card_tensor=processor.image_processor(images=[card],return_tensors='pt')
 n=card_tensor['pixel_values'].shape[0]
 assert torch.equal(card_tensor['pixel_values'],result.multi_modal_inputs['pixel_values'][:n])
 assert torch.equal(card_tensor['image_grid_thw'][0],result.multi_modal_inputs['image_grid_thw'][0])
 prefix=result.prompt_ids[0][-len(manager.calls[0]['ids']):].tolist()
 assert prefix==manager.calls[0]['ids']
 assert audit['distillation_separate_window']
 assert result.extra_fields['traj_success']
 card.save(OUT/'actual_card.png')
 from vagen.skills.provider import ViSkillProvider
 empty=OmegaConf.to_container(skill,resolve=True);empty['library']['path']=str(OUT/'empty_library')
 GymAgentLoop.skill_provider=ViSkillProvider(empty)
 class DistillManager:
  calls=0
  async def generate(self,request_id,prompt_ids,sampling_params,image_data=None,**kwargs):
   self.calls+=1
   text=('<think>Fixture.</think><answer>Down</answer>' if self.calls==1 else
    '{"when":{"goal_direction":"down","goal_manhattan_distance":1,"requires_clear_route":true},"what":[{"action":"down","count":1}]}')
   return SimpleNamespace(token_ids=processor.tokenizer.encode(text,add_special_tokens=False),log_probs=None)
 dm=DistillManager();handle.server_manager=dm
 distill=await AgentLoopWorkerBase._run_agent_loop(handle,{'temperature':0,'max_new_tokens':512},
  {'step':0,'sample_index':1,'rollout_n':0,'validate':False},agent_name='p0_fixture',env_name='FrozenLake',config=env,seed=6,max_turns=1,response_length_per_turn=512)
 assert dm.calls==2 and distill.extra_fields['distillation_audit']['valid']
 assert distill.extra_fields['distillation_audit']['model_calls']==1
 assert distill.extra_fields['mechanism_audit']['distillation_separate_window']
 assert 'goal_direction' not in processor.tokenizer.decode(distill.response_ids[0],skip_special_tokens=True)
 receipt={'fixture_only':True,'policy_calls':0,'real_worker_postprocessing':True,'native_training_kernel_executed':False,
  'same_card_object':True,'same_card_processed_pixels':True,'initial_prompt_identical':True,
  'all_image_tokens_loss_mask_zero':True,'generated_and_observation_spans_correct':True,
  'distillation_separate_window':True,'actual_distillation_fixture_excluded_from_ppo':True,'calls':len(manager.calls),'card_size':card.size,'card_patches':n,
  'reward':result.reward_score,'skill_bonus':audit['skill_bonus'],'audit':audit}
 (OUT/'result.json').write_text(json.dumps(jsonable(receipt),indent=2))
 print(json.dumps({k:v for k,v in receipt.items() if k!='audit'}),flush=True)
asyncio.run(main())
