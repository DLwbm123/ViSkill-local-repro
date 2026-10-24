"""Real processor/loop with explicitly scripted CPU fixture, zero model requests."""
import asyncio,json,os
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from transformers import AutoProcessor
from vagen.agent_loop.card_control_agent import CardControlAgent
ROOT=Path(os.environ['TASK_ROOT']);NAME='card_content_controls_v1';OUT=ROOT/'artifacts/component_tests'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());DATA=json.loads((OUT/'data_manifest.json').read_text())
async def main():
 proc=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True);captures={}
 for arm in P['arms']:
  folder=OUT/'cpu_fixture'/arm;folder.mkdir(parents=True,exist_ok=False)
  skill=OmegaConf.to_container(OmegaConf.load(ROOT/'src/configs/low_resource/directional_skill_system.yaml').skill_system,resolve=True)
  skill['library']={'path':str(folder/'empty') if arm=='N' else str(ROOT/P['source_library']),'max_size':32};skill['reward']['guided_weight']=0
  config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'multi_turn':{},'prompt_length':8192,'response_length':4096}},'data':{'apply_chat_template_kwargs':{}},'env_registry':{'FrozenLake':'vagen.envs.frozenlake.frozenlake_env.FrozenLake'},'skill_system':skill})
  class ScriptedManager:
   async def generate(self,**kw):
    assert arm not in captures
    images=kw.get('image_data') or []
    captures[arm]={'prompt_tokens':len(kw['prompt_ids']),'image_count':len(images),'image_sizes':[list(x.size) for x in images]}
    if arm=='Blank':assert all(v==(255,255) for v in images[0].getextrema())
    if arm in ['V','Blank']:
     pixels=proc.image_processor(images=[images[0]],return_tensors='pt');captures[arm]['reference_grid']=pixels['image_grid_thw'].tolist()
    return SimpleNamespace(token_ids=proc.tokenizer.encode('<think>Explicit scripted CPU fixture.</think><answer>Up, Up, Up</answer>',add_special_tokens=False),log_probs=None)
  cls=type('Fixture'+arm,(CardControlAgent,),{'_class_initialized':False,'_skill_initialized':False})
  agent=cls(SimpleNamespace(config=config),ScriptedManager(),proc.tokenizer,proc);agent.card_condition=arm;agent.card_reference_text=DATA['text_reference']
  os.environ['PILOT_OUT']=str(folder);os.environ['TASK_PHASE']='A' if arm=='N' else 'B';os.environ['PILOT_DATA_MANIFEST']=str(OUT/'data_manifest.json');os.environ['MECHANISM_AUDIT_DIR']=str(folder/'audit')
  result=await agent.run(P['sampling'].copy(),env_name='FrozenLake',config=P['environment'],seed=DATA['splits']['dev'][0]['seed'],max_turns=5,response_length_per_turn=512,pilot_phase='dev',pilot_policy_version=0,validate=True)
  assert result.num_turns==1 and not result.extra_fields['distillation_audit']['model_calls']
  if arm=='Text':assert agent.card_reference_audit['obs_str'].endswith(DATA['text_reference'])
 assert captures['V']['image_count']==captures['Blank']['image_count']==2
 assert captures['N']['image_count']==captures['Text']['image_count']==1
 assert captures['V']['prompt_tokens']==captures['Blank']['prompt_tokens']
 assert captures['V']['reference_grid']==captures['Blank']['reference_grid']
 output={'status':'PASS','model_calls':0,'scripted_fixture_generate_calls':4,'real_processor_and_loop':True,'captures':captures}
 (OUT/'cpu_control_checks.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
asyncio.run(main())
