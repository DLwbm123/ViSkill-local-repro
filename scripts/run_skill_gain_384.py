"""Serial frozen b0 content controls under two predeclared engine random streams."""
import os,json,time,runpy,asyncio,socket,subprocess
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from transformers import AutoProcessor
from vagen.skill_gain_probe import ROOT,SRC,OUT,P,GainLoop,Manager,prepare_output_folders
from vagen.official_train_observer import plain,write
async def run_repeat(engine,repeat,processor):
    cfg=runpy.run_path(str(SRC/'scripts/official_train_config.py'))['config']();cfg.skill_system.distill.mode='disabled';cfg.skill_system.reward.mode='success';cfg.skill_system.cold_start.enable=False
    data=json.loads((OUT/'data_manifest.json').read_text());items={r['seed']:r for r in data['records']};manager=Manager(engine,processor,repeat);agents={}
    for condition in P['conditions']:
        conf=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));conf.skill_system.library.path=str(OUT/'empty_library') if condition=='N' else str(ROOT/P['source_library'])
        cls=type(f'Frozen{condition}{repeat}',(GainLoop,),{'_class_initialized':False,'_skill_initialized':False});agent=cls(SimpleNamespace(config=conf),manager,processor.tokenizer,processor);agent.condition=condition;agents[condition]=agent
    for position,entry in enumerate(data['schedule']):
        if entry['repeat']!=repeat:continue
        assert time.time()<P['cooperative_stop_unix'] and not (OUT/'STOP_REQUESTED.json').exists()
        seed=entry['seed'];condition=entry['condition'];item=items[seed];episode_id=f'r{repeat}_{condition}_{seed}';manager.item=item;manager.condition=condition;manager.episode_id=episode_id;manager.turn=0
        path=OUT/'episodes'/f'{episode_id}.json';assert not path.exists()
        write(OUT/'CURRENT_EPISODE.json',dict(entry,schedule_position=position,started_unix=time.time()))
        result=await agents[condition].run(P['sampling'].copy(),env_name='FrozenLake',config=P['environment'],seed=seed,max_turns=5,response_length_per_turn=512,validate=True)
        extra=result.extra_fields;expected=item['skill_id'] if condition!='N' else None;assert extra['retrieved_skill_id']==expected and plain(extra['initial_env_state'])==item['state']
        assert not extra['distilled_strategy'] and result.num_turns==manager.turn
        record={k:plain(extra.get(k)) for k in ['reward_extra_info','traj_success','retrieved_skill_id','skill_score','initial_env_state','trajectory_actions','trajectory_positions','distilled_strategy']};record.update(repeat=repeat,condition=condition,seed=seed,stratum=item['stratum'],episode_id=episode_id,num_turns=result.num_turns,response_ids=result.response_ids,response_mask=result.response_mask,completed_unix=time.time())
        folder=OUT/'trajectory_images'/episode_id;folder.mkdir()
        record['trajectory_images']=[]
        for i,img in enumerate(extra.get('trajectory_frames',[])):
            p=folder/f'{i:02d}.png';img.save(p);record['trajectory_images'].append(str(p.relative_to(OUT)))
        write(path,record)
        print('EPISODE',repeat,condition,seed,bool(extra['traj_success']),flush=True)
    assert json.loads((ROOT/P['source_library']/'skill_index.json').read_text())==data['library_index']

def selected_repeat():
    repeat=int(os.environ.get('TASK_REPEAT_INDEX','0'));assert repeat in [0,1]
    existing=[json.loads(p.read_text()) for p in (OUT/'episodes').glob('*.json')]
    if repeat==0:assert not existing and not (OUT/'dispatch_count.json').exists()
    else:
        data=json.loads((OUT/'data_manifest.json').read_text());expected={(x['seed'],x['repeat'],x['condition']) for x in data['schedule'] if x['repeat']==0}
        assert len(existing)==192 and {(x['seed'],x['repeat'],x['condition']) for x in existing}==expected
        events=[json.loads(x) for x in (OUT/'requests.jsonl').read_text().splitlines()]
        assert all(x['repeat']==0 for x in events) and not [x for x in events if x['event']=='error']
        assert len([x for x in events if x['event']=='intent'])==len([x for x in events if x['event']=='result'])==json.loads((OUT/'dispatch_count.json').read_text())['count']
        assert not (OUT/'engine_args_r1.json').exists()
    return repeat

def launch():
    import sglang
    assert P['status']=='FROZEN_CPU_READY';selected=selected_repeat()
    prepare_output_folders()
    processor=AutoProcessor.from_pretrained(ROOT/'models/b0',local_files_only=True)
    for repeat,seed in [(selected,P['engine_seeds'][selected])]:
        assert time.time()<P['cooperative_stop_unix'];free=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=60000
        loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);engine=None
        try:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            args=dict(model_path=str(ROOT/'models/b0'),host='127.0.0.1',port=port,dtype='bfloat16',tp_size=1,mem_fraction_static=.30,context_length=9000,max_running_requests=1,disable_cuda_graph=True,attention_backend='triton',mm_attention_backend='sdpa',trust_remote_code=False,random_seed=seed,log_level='info')
            write(OUT/f'engine_args_r{repeat}.json',args);engine=sglang.Engine(**args);asyncio.set_event_loop(loop);loop.run_until_complete(run_repeat(engine,repeat,processor))
        finally:
            if engine is not None:engine.shutdown()
            loop.close()
    assert len(list((OUT/'episodes').glob('*.json')))==192*(selected+1)
    write(OUT/f'REPEAT_{selected}_COMPLETED.json',{'repeat':selected,'episodes':192,'completed_unix':time.time(),'engine_process_completed':True})
    if selected==0:return
    write(OUT/'WORKER_COMPLETED.json',{'status':'COMPLETE_384_FROZEN_SKILL_GAIN_EPISODES','completed_unix':time.time(),'episodes':384,'model_calls':json.loads((OUT/'dispatch_count.json').read_text())['count'],'optimizer_steps':0,'admissions':0,'distill_calls':0,'utility_updates':0})
if __name__=='__main__':launch()
