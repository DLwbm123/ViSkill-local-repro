"""Complete/early-stop paired context summary, no inference or test access."""
import json,os
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);NAME='fixed_library_ppo_v1';OUT=ROOT/'results/ppo_smoke'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());summaries={};allrows=[];calls=0
source_library=json.loads((ROOT/P['source_library']/'skill_index.json').read_text())
for arm in ['A','B']:
 folder=OUT/arm;assert json.loads((folder/'context_libraries/cards/skill_index.json').read_text())==source_library,'fixed library changed';assert not (folder/'context_libraries/empty/skill_index.json').exists();complete=json.loads((folder/'COMPLETE.json').read_text());c=json.loads((folder/'completion/trainer.json').read_text())
 episode_files=list((folder/'episodes').glob('*/episode.json'));episodes=[json.loads(p.read_text()) for p in episode_files];calls_arm=int((folder/'model_calls.txt').read_text());saved=len(list((folder/'requests').glob('*.json')));assert calls_arm==saved<=P['max_model_calls_per_arm'];calls+=calls_arm
 head=json.loads((folder/'initialization/critic_rank0.json').read_text());counts={}
 for version in c['cross_versions']:
  counts[str(version)]={}
  for context in ['empty','cards']:
   counts[str(version)][context]={}
   for group in ['natural','conditional']:
    r=[x for x in episodes if x['phase']=='dev' and x['policy_version']==version and x['fixed_context']==context and x['fixed_group']==group]
    assert len(r)==(64 if group=='natural' else 32)
    counts[str(version)][context][group]={'episodes':len(r),'success':sum(x['traj_success'] for x in r),'quality':sum(x['traj_success'] and len(x['trajectory_actions'])<=x['bfs'] for x in r),'multi_success':sum(x['traj_success'] and x['bfs']>1 for x in r),'multi_quality':sum(x['traj_success'] and x['bfs']>1 and len(x['trajectory_actions'])<=x['bfs'] for x in r)}
 for path,ep in zip(episode_files,episodes):
  row={k:ep.get(k) for k in ['phase','policy_version','fixed_context','fixed_group','category','initial_env_state','bfs','traj_success','trajectory_actions','retrieved_skill_id','terminal_reason','reward_support','distillation_audit']};row.update(arm=arm,quality=bool(ep['traj_success'] and len(ep['trajectory_actions'])<=ep['bfs']));row['seed']=int(path.parent.name.split('_')[2]);allrows.append(row)
 updates=[json.loads(line) for p in (folder/'update_audit').glob('*.jsonl') for line in p.read_text().splitlines()];assert all(x['finite'] for x in updates)
 summaries[arm]={'completion':c,'dev':counts,'model_calls':calls_arm,'saved_requests':saved,'episodes':len(episodes),'optimizer_audit_rows':len(updates),'optimizer_all_finite':True,'optimizer_sample_deltas_nonzero':all(x['sample_delta_l2']>0 for x in updates),'cross_versions':c['cross_versions']}
assert calls<=P['max_model_calls_pair']
head_equal=json.loads((OUT/'A/initialization/critic_rank0.json').read_text())['head_values']==json.loads((OUT/'B/initialization/critic_rank0.json').read_text())['head_values'];assert head_equal
paired={};gate=False
if all(summaries[a]['completion']['training_batches']==4 for a in ['A','B']):
 def rows(arm,version,group):return {r['seed']:r for r in allrows if r['arm']==arm and r['phase']=='dev' and r['policy_version']==version and r['fixed_context']=='cards' and r['fixed_group']==group}
 b=rows('B',4,'conditional')
 for name,c in [('B_self',rows('B',0,'conditional')),('B_vs_A',rows('A',4,'conditional'))]:
  paired[name]={}
  for metric in ['traj_success','quality']:
   up=sum(b[s][metric] and not c[s][metric] for s in b);down=sum(c[s][metric] and not b[s][metric] for s in b);paired[name][metric]={'improved':up,'worsened':down,'net':up-down}
 before=summaries['B']['dev']['0']['cards']['natural'];after=summaries['B']['dev']['4']['cards']['natural']
 gate=all(v['net']>=4 and v['worsened']<=2 for comparison in paired.values() for v in comparison.values()) and before['multi_success']-after['multi_success']<4 and before['multi_quality']-after['multi_quality']<4
result={'summary':summaries,'paired_conditional_same_library':paired,'paired_critic_head_equal':head_equal,'fixed_libraries_match_original':True,'next_investment_gate':gate,'stable_learning_benefit_established':False,'total_model_calls':calls,'total_episodes':len(allrows),'checkpoint_saved':False,'independent_recovery_verified':False,'records':allrows}
(OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
