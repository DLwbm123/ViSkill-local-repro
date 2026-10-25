"""Real processor/native loop fixtures for all control conditions, no model calls."""
import os,json,runpy,asyncio,subprocess,time
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from transformers import AutoProcessor
import torch,ray
from vagen.skill_gain_probe import ROOT,SRC,OUT,P,GainLoop
from vagen.official_train_observer import plain,write
async def main():
    data=json.loads((OUT/'data_manifest.json').read_text());item=next(r for r in data['records'] if r['eligible']);proc=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True);cfg=runpy.run_path(str(SRC/'scripts/official_train_config.py'))['config']();cfg.skill_system.distill.mode='disabled';cfg.skill_system.reward.mode='success';captures={}
    for condition in P['conditions']:
        conf=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));conf.skill_system.library.path=str(OUT/'empty_library') if condition=='N' else str(ROOT/P['source_library'])
        class Fixture:
            async def generate(self,**kw):
                if condition not in captures:
                    images=kw.get('image_data') or [];captures[condition]={'prompt_ids':kw['prompt_ids'],'image_sizes':[list(x.size) for x in images],'grids':proc.image_processor(images=images,return_tensors='pt')['image_grid_thw'].tolist()}
                    if condition=='Blank':assert all(x==(255,255) for x in images[0].getextrema())
                return SimpleNamespace(token_ids=proc.tokenizer.encode('<think>Explicit CPU fixture.</think><answer>Stop</answer>',add_special_tokens=False),log_probs=None)
        cls=type('Fixture'+condition,(GainLoop,),{'_class_initialized':False,'_skill_initialized':False});agent=cls(SimpleNamespace(config=conf),Fixture(),proc.tokenizer,proc);agent.condition=condition
        out=await agent.run(P['sampling'].copy(),env_name='FrozenLake',config=P['environment'],seed=item['seed'],max_turns=1,response_length_per_turn=512,validate=True)
        assert out.num_turns==1 and not out.extra_fields['distilled_strategy'] and plain(out.extra_fields['initial_env_state'])==item['state']
        assert out.extra_fields['retrieved_skill_id']==(item['skill_id'] if condition!='N' else None)
    assert captures['V']['prompt_ids']==captures['Blank']['prompt_ids'] and captures['V']['image_sizes']==captures['Blank']['image_sizes'] and captures['V']['grids']==captures['Blank']['grids'];assert len(captures['N']['image_sizes'])==1 and len(captures['V']['image_sizes'])==2
    assert not ray.is_initialized() and not torch.cuda.is_initialized()
    result={'status':'CPU_READY','completed_unix':time.time(),'actual_model_calls':0,'GPU_calls':0,'scripted_fixture_episodes':3,'V_Blank_first_prompt_IDs_sizes_grids_equal':True,'Blank_reference_pixels_white':True,'N_no_card_V_Blank_original_retrieval_id_correct':True,'original_state_matches_frozen_manifest':True,'distillation_disabled':True,'captures':{k:{'prompt_token_count':len(v['prompt_ids']),'image_sizes':v['image_sizes'],'grids':v['grids']} for k,v in captures.items()}}
    write(OUT/'cpu_check.json',result);print(json.dumps(result))
asyncio.run(main())
