"""Read completed frozen capacity evidence; no calls, writes to source, or gate changes."""
import argparse,json
from collections import Counter
from pathlib import Path
from vagen.skills.directional_contract import parse_directional_contract,displacement_condition

def analyze(root):
 root=Path(root);result=json.loads((root/'result.json').read_text())
 summaries=result['summaries'];assert len(summaries)==128
 expected=list(range(20001,20129));assert [s['seed'] for s in summaries]==expected
 rows=[json.loads((root/f'episode_{seed}/episode.json').read_text()) for seed in expected]
 assert all(row['retrieved_skill_id'] is None for row in rows)
 admission={}
 for p in (root/'audit/admissions').glob('*.json'):
  for row in json.loads(p.read_text())['rows']:
   assert row['seed'] not in admission;admission[row['seed']]=row
 assert sorted(admission)==expected
 from vagen.skills.provider import ViSkillProvider
 from vagen.skills.reward import compute_reward
 provider=ViSkillProvider({'library':{'path':str(root/'diagnostic_library')},
  'distill':{'representation':'directional_contract_v1','allow_assistant_envelope':True},
  'retrieval':{'similarity_weight':.9,'min_score':.8,'contract_guard':True}})
 provider.init_env_type('FrozenLake')
 counterfactual=[]
 for row in rows:
  row['initial_env_state']['player_pos']=tuple(row['initial_env_state']['player_pos'])
  skill=provider.retrieve(row['initial_env_state'])
  if skill and skill['score']>=.8:
   bonus=compute_reward(env_rewards=[],mode='state_progress_bounded',traj_success=row['traj_success'],
    skill_trajectory=skill['trajectory'],model_trajectory=row['trajectory_actions'],guided_weight=.8,
    reference_positions=skill['positions'],model_positions=row['trajectory_positions'])
   counterfactual.append({'seed':row['seed'],'success':row['traj_success'],'skill_id':skill['id'],
    'reference_actions':len(skill['trajectory']),'score':skill['score'],'posthoc_bonus':bonus})
 failures=Counter();semantic=Counter();details=[]
 for summary,row in zip(summaries,rows):
  audit=row.get('distillation_audit') or {};raw=audit.get('raw');stages={}
  if audit.get('model_calls'):
   try:
    contract,actions=parse_directional_contract(raw,allow_assistant_envelope=True)
    positions=row['trajectory_positions'];dy,dx=[b-a for a,b in zip(positions[0],positions[-1])]
    direction,distance=displacement_condition(dy,dx)
    stages={'parse_valid':True,'direction_correct':contract['when']['goal_direction']==direction,
     'distance_correct':contract['when']['goal_manhattan_distance']==distance,
     'actions_exact':actions==row['trajectory_actions']}
    for key,value in stages.items():semantic[key]+=int(value)
   except (ValueError,TypeError,KeyError) as e:stages={'parse_valid':False,'error':str(e)}
   if not audit.get('valid'):failures[audit.get('reason','unknown')]+=1
  details.append({'seed':row['seed'],'bfs':summary['bfs'],'actions':row['trajectory_actions'],
   'success':row['traj_success'],'quality_eligible':bool(row['traj_success'] and summary['bfs'] and len(row['trajectory_actions'])/summary['bfs']<=1),
   'distillation_called':bool(audit.get('model_calls')),'distillation_valid':bool(audit.get('valid')),
   'stage_checks':stages,'admission':admission[row['seed']]['admission_audit']})
 count=lambda key:sum(d[key] for d in details)
 stages={'episodes':128,'successful':count('success'),'quality_eligible':count('quality_eligible'),
  'distillation_called':count('distillation_called'),'distillation_valid':count('distillation_valid'),
  'admitted':sum(d['admission']=='admitted' for d in details)}
 assert stages['distillation_called']==result['distillation_calls']
 assert stages['admitted']==result['admitted']
 assert stages['distillation_valid']==result['valid_distillations']
 multi=[d for d in details if d['bfs'] and d['bfs']>=2]
 multi_counts={'episodes':len(multi),'successful':sum(d['success'] for d in multi),
  'quality_eligible':sum(d['quality_eligible'] for d in multi),'distillation_valid':sum(d['distillation_valid'] for d in multi),
  'admitted':sum(d['admission']=='admitted' for d in multi)}
 return {'posthoc_no_injection_counterfactual':{'eligible':len(counterfactual),'failure_positive_bonus':sum(not r['success'] and r['posthoc_bonus']>0 for r in counterfactual),'rows':counterfactual,'actual_skill_injection':False,'actual_training':False},'funnel':stages,'multi_step_bfs_ge2':multi_counts,'distillation_failures':dict(failures),
  'semantic_checks_among_called':dict(semantic),'admission_reasons':dict(Counter(d['admission'] for d in details)),
  'strata':result['strata'],'unique_exact_boards':result['unique_exact_boards'],
  'admitted_action_lengths':result['admitted_action_lengths'],'details':details}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
 print(json.dumps(analyze(args.root),indent=2))
