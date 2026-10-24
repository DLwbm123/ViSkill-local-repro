"""Read-only initial greedy case comparison, no model calls."""
import json,os
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);out=ROOT/'results/ppo_smoke/fixed_library_ppo_v1';baseline={};counts={}
for arm in ['A','B']:
 rows=[]
 for path in (out/arm/'episodes').glob('dev_0_*/episode.json'):
  d=json.loads(path.read_text());rows.append({'seed':int(path.parent.name.split('_')[2]),'context':d['fixed_context'],'group':d['fixed_group'],'success':d['traj_success'],'quality':bool(d['traj_success'] and len(d['trajectory_actions'])<=d['bfs']),'actions':d['trajectory_actions']})
 baseline[arm]={(x['seed'],x['context']):x for x in rows};assert len(baseline[arm])==192
 counts[arm]={c:{g:{'episodes':len(r:=[x for x in rows if x['context']==c and x['group']==g]),'success':sum(x['success'] for x in r),'quality':sum(x['quality'] for x in r)} for g in ['natural','conditional']} for c in ['empty','cards']}
assert baseline['A'].keys()==baseline['B'].keys();keys=baseline['A'].keys()
v={'model_calls':0,'episodes_per_arm':192,**{m+'_match':sum(baseline['A'][k][m]==baseline['B'][k][m] for k in keys) for m in ['success','quality','actions']},'counts':counts,'mismatches':[{'A':baseline['A'][k],'B':baseline['B'][k]} for k in keys if baseline['A'][k]['success']!=baseline['B'][k]['success'] or baseline['A'][k]['quality']!=baseline['B'][k]['quality']],'scope':'Observed initial greedy cases only; paired random critic heads do not imply identical policy outputs or training stochastic streams.'}
(out/'baseline_pair_review.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
