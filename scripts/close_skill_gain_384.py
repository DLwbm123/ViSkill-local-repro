"""Seal all384 paired outcomes; the preregistered count gate is not a significance test."""
import os,json,time,random,shutil,subprocess
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='skill_gain_384_v1';OUT=ROOT/'artifacts/reproduction'/NAME;STATE=ROOT/'state/reproduction';P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text())
def read(p):return json.loads(p.read_text())
def investment_gate(rows,items):
    by={(r['seed'],r['repeat'],r['condition']):r for r in rows};benefit=[];harm=[]
    for item in items:
        seed=item['seed']
        if item['eligible'] and all(by[seed,k,'V']['traj_success'] and not by[seed,k,'N']['traj_success'] for k in [0,1]):benefit.append(seed)
        if any(by[seed,k,'N']['traj_success'] and not by[seed,k,'V']['traj_success'] for k in [0,1]):harm.append(seed)
    patterns=sorted({tuple(item['action_pattern']) for item in items if item['seed'] in benefit});checks={'reproducible_benefit_maps_ge12':len(benefit)>=12,'any_repeat_harm_maps_le4':len(harm)<=4,'distinct_action_patterns_ge3':len(patterns)>=3,'contains_multistep_pattern':any(len(x)>1 for x in patterns)}
    return {'pass':all(checks.values()),'checks':checks,'benefit_seeds':benefit,'harm_seeds':harm,'benefit_action_patterns':patterns,'scope':'Investment count gate; paired64maps/two repeats, not384independent maps; Blank comparison reported separately without adding a posthoc threshold'}
def main():
    from vagen.skills.solver import FrozenLakeSolver
    from vagen.envs.frozenlake.utils.utils import parse_response
    end=read(STATE/f'{NAME}_EXIT.json');assert end['exit_code']==0 and end['reason'] is None and not end['argv_violations'];assert read(OUT/'WORKER_COMPLETED.json')['episodes']==384
    data=read(OUT/'data_manifest.json');items=data['records'];rows=[read(p) for p in (OUT/'episodes').glob('*.json')];assert len(rows)==384
    expected={(e['seed'],e['repeat'],e['condition']) for e in data['schedule']};by={(r['seed'],r['repeat'],r['condition']):r for r in rows};assert len(by)==384 and set(by)==expected
    events=[json.loads(x) for x in (OUT/'requests.jsonl').read_text().splitlines()];requests=[x for x in events if x['event']=='result'];intents=[x for x in events if x['event']=='intent'];assert not [x for x in events if x['event']=='error'];assert len(requests)==len(intents)==read(OUT/'dispatch_count.json')['count']<=1920 and len({x['number'] for x in requests})==len(requests);assert sum(x['num_turns'] for x in rows)==len(requests)
    first={(r['seed'],r['repeat'],r['condition']):r for r in requests if r['first_request']};assert set(first)==expected
    for item in items:
        seed=item['seed'];state=dict(item['state'],player_pos=tuple(item['state']['player_pos']));distance=FrozenLakeSolver().distance(state)
        for repeat in [0,1]:
            v=first[seed,repeat,'V'];b=first[seed,repeat,'Blank'];assert v['prompt_ids']==b['prompt_ids'] and v['image_sizes']==b['image_sizes'] and v['image_grid_thw']==b['image_grid_thw']
            for condition in P['conditions']:
                r=by[seed,repeat,condition];assert r['initial_env_state']==item['state'] and not r['distilled_strategy'];assert r['retrieved_skill_id']==(item['skill_id'] if condition!='N' else None)
                r['initial_BFS_distance']=distance;r['shortest_path_quality']=bool(r['traj_success'] and distance and len(r['trajectory_actions'])<=distance)
                last=r['trajectory_positions'][-1];r['hole']=r['initial_env_state']['desc'][last[0]][last[1]]=='H'
                own=[q for q in requests if q['episode_id']==r['episode_id']];actions=[parse_response(q['raw'],prompt_format='free_think',max_actions=5)['actions'] for q in own];r['stop_turns']=sum('stop' in a for a in actions);r['unsupported_turns']=sum(any(x not in ['up','down','left','right'] for x in a) for a in actions)
    gate=investment_gate(rows,items);summary={};paired={}
    for stratum in ['natural','retrievable','balanced_panel']:
        selected=[r for r in rows if stratum=='balanced_panel' or r['stratum']==stratum]
        summary[stratum]={c:{'episodes':len([r for r in selected if r['condition']==c]),**{k:sum(r[k] for r in selected if r['condition']==c) for k in ['traj_success','shortest_path_quality','hole','stop_turns','unsupported_turns']}} for c in P['conditions']}
        seeds=[r['seed'] for r in items if stratum=='balanced_panel' or r['stratum']==stratum];paired[stratum]={}
        for treatment,control in [('V','N'),('V','Blank')]:
            values=[sum(int(by[s,k,treatment]['traj_success'])-int(by[s,k,control]['traj_success']) for k in [0,1])/2 for s in seeds];rng=random.Random(20261025);draws=sorted(sum(values[rng.randrange(len(values))] for _ in values)/len(values) for _ in range(10000))
            paired[stratum][treatment+'-'+control]={'paired_map_mean_difference':sum(values)/len(values),'map_cluster_bootstrap95':[draws[249],draws[9749]],'repeat_specific_improved':[sum(by[s,k,treatment]['traj_success'] and not by[s,k,control]['traj_success'] for s in seeds) for k in [0,1]],'repeat_specific_regressed':[sum(by[s,k,control]['traj_success'] and not by[s,k,treatment]['traj_success'] for s in seeds) for k in [0,1]]}
    receipt=read(STATE/f'{NAME}_receipt.json');residual=[]
    for q in psutil.process_iter(['pid','create_time','uids']):
        try:
            if q.info['uids'].real==os.getuid() and q.info['create_time']>=receipt['start_unix']-1 and q.environ().get('TASK_RUN_TOKEN')==receipt['run_token'] and q.status()!=psutil.STATUS_ZOMBIE:residual.append(q.pid)
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not residual and read(ROOT/P['source_library']/'skill_index.json')==data['library_index']
    result={'status':'COMPLETE_384_SKILL_GAIN','closed_unix':time.time(),'episodes':384,'unique_maps':64,'repeats':2,'generation_requests':len(requests),'request_errors':0,'output_tokens':sum(len(q['output_ids']) for q in requests),'summary':summary,'paired':paired,'investment_gate':gate,'recommendation':'WORTH_NEXT_INTERNALIZATION_PILOT' if gate['pass'] else 'DO_NOT_START_INTERNALIZATION_WITH_CURRENT_SKILL_SUPPORT','complete_records':rows,'library_unchanged':True,'new_optimizer_steps':0,'new_distill_calls':0,'new_admissions':0,'new_utility_updates':0,'live_owned_non_zombie':residual,'exit':end,'limits':['Two greedy reinitializations do not imply128independent maps or strict bitwise repeatability','Balanced32natural/32retrievable panel is not a natural population estimate','Count gate is for continued investment, not significance or learned-policy improvement','No automatic internalization training launched']}
    (OUT/'result.json').write_text(json.dumps(result,indent=2));dest=OUT/'public_bundle';assert not dest.exists();dest.mkdir()
    for name in ['result.json','data_manifest.json','cpu_check.json','WORKER_COMPLETED.json','engine_args_r0.json','engine_args_r1.json']:
        shutil.copy2(OUT/name,dest/name)
    (dest/'requests.jsonl').write_text('\n'.join(json.dumps(dict({k:v for k,v in q.items() if k not in ['prompt_ids','images']},prompt_token_count=len(q['prompt_ids']),bulk_inputs_retained_remote=True)) for q in requests)+'\n')
    receipt.pop('run_token');(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));(dest/'EXIT.json').write_text(json.dumps(end,indent=2))
    usage=sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in [ROOT,Path('/data/envs/z0'),Path('/data/tools/z0'),Path('/data/tmp/z0')]);assert usage<receipt['strict_hard_bytes'] and end['end_unix']<P['deadline_unix'];result.update(final_four_root_bytes=usage,new_bytes_from_original_experiment_baseline=usage-receipt['baseline_bytes'])
    for p in [OUT/'result.json',dest/'result.json']:p.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['complete_records','limits']},indent=2))
if __name__=='__main__':main()
