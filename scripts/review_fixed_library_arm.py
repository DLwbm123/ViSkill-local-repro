"""Completed-arm evidence review; no inference, updates, or protocol changes."""
import os,json,random
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);arm=os.environ['TASK_REVIEW_ARM'];assert arm in ['A','B']
folder=ROOT/'results/ppo_smoke/fixed_library_ppo_v1'/arm
complete=json.loads((folder/'COMPLETE.json').read_text());completion=json.loads((folder/'completion/trainer.json').read_text())
records=[]
for p in (folder/'episodes').glob('*/episode.json'):
 d=json.loads(p.read_text());records.append({'seed':int(p.parent.name.split('_')[2]),**{k:d.get(k) for k in ['phase','policy_version','fixed_context','fixed_group','category','bfs','traj_success','trajectory_actions','retrieved_skill_id','terminal_reason','reward_support']},'quality':bool(d['traj_success'] and len(d['trajectory_actions'])<=d['bfs'])})
summary={};paired={};versions=completion['cross_versions'];final=versions[-1]
for version in versions:
 summary[str(version)]={}
 for context in ['empty','cards']:
  summary[str(version)][context]={}
  for group in ['natural','conditional']:
   rows=[r for r in records if r['phase']=='dev' and r['policy_version']==version and r['fixed_context']==context and r['fixed_group']==group]
   assert len(rows)==(64 if group=='natural' else 32)
   multi=[r for r in rows if r['bfs']>1]
   summary[str(version)][context][group]={'episodes':len(rows),'success':sum(r['traj_success'] for r in rows),'quality':sum(r['quality'] for r in rows),'multi_episodes':len(multi),'multi_success':sum(r['traj_success'] for r in multi),'multi_quality':sum(r['quality'] for r in multi)}
for context in ['empty','cards']:
 paired[context]={}
 for group in ['natural','conditional']:
  before={r['seed']:r for r in records if r['phase']=='dev' and r['policy_version']==0 and r['fixed_context']==context and r['fixed_group']==group};after={r['seed']:r for r in records if r['phase']=='dev' and r['policy_version']==final and r['fixed_context']==context and r['fixed_group']==group}
  assert before.keys()==after.keys();paired[context][group]={}
  for metric in ['traj_success','quality']:
   deltas=[int(after[s][metric])-int(before[s][metric]) for s in sorted(before)];n=len(deltas);rng=random.Random(20261024)
   draws=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
   paired[context][group][metric]={'improved':sum(x==1 for x in deltas),'worsened':sum(x==-1 for x in deltas),'net':sum(deltas),'difference':sum(deltas)/n,'posthoc_descriptive_paired_map_95':[draws[249],draws[9749]]}
updates=[json.loads(line) for p in (folder/'update_audit').glob('*.jsonl') for line in p.read_text().splitlines()]
assert all(r['finite'] for r in updates)
source=ROOT/'results/ppo_smoke/bootstrap_card_pair_v1/B/library/skill_index.json';assert json.loads((folder/'context_libraries/cards/skill_index.json').read_text())==json.loads(source.read_text())
train=[r for r in records if r['phase']=='train'];assert len(train)==completion['training_batches']*32
result={'arm':arm,'complete':complete,'completion':completion,'total_episodes':len(records),'saved_requests':len(list((folder/'requests').glob('*.json'))),'summary':summary,'paired_self_baseline':paired,'train_episodes':len(train),'train_success':sum(r['traj_success'] for r in train),'train_quality':sum(r['quality'] for r in train),'train_skill4_exposures':sum(r['retrieved_skill_id']=='skill_4' for r in train),'optimizer_audit_rows':len(updates),'optimizer_all_finite':True,'optimizer_sampled_deltas_nonzero':all(r['sample_delta_l2']>0 for r in updates),'library_matches_source':True,'uncertainty_scope':'Descriptive conditional on this one observed run; not training-seed or natural-population confirmation.','records':records}
(folder/'arm_review_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
