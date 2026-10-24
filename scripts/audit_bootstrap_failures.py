"""Read-only recomputation of published outcomes and WHEN/WHAT failure split."""
import json
from pathlib import Path
from vagen.skills.directional_contract import parse_directional_contract,validate_directional_observation,displacement_condition
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/evidence/components/bootstrap_card_pair_v1'
rows=json.loads((BASE/'all_episode_outcomes.json').read_text());dist=json.loads((BASE/'all_distillation_outputs.json').read_text())
strata={}
for length in [1,2,3,4]:
 label=str(length) if length<4 else '4+'
 def filtered(arm,version):return [r for r in rows if r['arm']==arm and r['phase']=='dev' and r['policy_version']==version and (r['bfs']==length if length<4 else r['bfs']>=4)]
 strata[label]={'maps':len(filtered('B',0)),'before':sum(r['traj_success'] for r in filtered('B',0)),'A_after':sum(r['traj_success'] for r in filtered('A',4)),'B_after':sum(r['traj_success'] for r in filtered('B',4))}
first={arm:{'episodes':len(x:=[r for r in rows if r['arm']==arm and r['phase']=='train' and r['policy_version']==0]),'success':sum(r['traj_success'] for r in x),'quality':sum(r['quality_eligible'] for r in x),'injected':sum(bool(r['retrieved_skill_id']) for r in x)} for arm in ['A','B']}
multi_before={r['seed']:r for r in rows if r['arm']=='B' and r['phase']=='dev' and r['policy_version']==0 and r['bfs']>1}
multi_after={r['seed']:r for r in rows if r['arm']=='B' and r['phase']=='dev' and r['policy_version']==4 and r['bfs']>1}
transitions={'improved':sum(multi_after[s]['traj_success'] and not r['traj_success'] for s,r in multi_before.items()),'worsened':sum(r['traj_success'] and not multi_after[s]['traj_success'] for s,r in multi_before.items()),'regression_seeds':[s for s,r in multi_before.items() if r['traj_success'] and not multi_after[s]['traj_success']]}
records=[]
for d in dist:
 if len(d['actions'])<=1:continue
 c,a=parse_directional_contract(d['audit']['raw'],allow_assistant_envelope=True)
 q={'arm':d['arm'],'seed':d['seed'],'phase':d['phase'],'policy_version':d['policy_version'],'original_valid':d['audit']['valid'],'predicted_distance':c['when']['goal_manhattan_distance'],'what_matches_actions':a==d['actions']}
 pos=d['positions'];dy,dx=[int(b)-int(a) for a,b in zip(pos[0],pos[-1])];direction,distance=displacement_condition(dy,dx)
 q.update(actual_direction=direction,actual_distance=distance)
 c['when']['goal_direction']=direction;c['when']['goal_manhattan_distance']=distance
 original=next(r for r in rows if all(r[k]==d[k] for k in ['arm','seed','phase','policy_version']));assert original['traj_success']
 observed={'traj_success':original['traj_success'],'trajectory_positions':pos,'trajectory_actions':d['actions'],'initial_env_state':d['initial_env_state']}
 try:validate_directional_observation(json.dumps(c),observed,allow_assistant_envelope=True);q['when_only_counterfactual_valid']=True
 except (ValueError,TypeError,KeyError) as exc:q['when_only_counterfactual_valid']=False;q['counterfactual_rejection']=str(exc)
 records.append(q)
failed=[r for r in records if not r['original_valid']]
result={'model_calls':0,'original_outputs_modified':False,'BFS_dev':strata,'first_training_batch':first,'B_multi_transitions':transitions,'distill_multistep':{'total':len(records),'valid':sum(r['original_valid'] for r in records),'failed':len(failed),'failed_distance_one':sum(r['predicted_distance']==1 for r in failed),'failed_what_matches':sum(r['what_matches_actions'] for r in failed),'failed_when_only_counterfactual_pass':sum(r['when_only_counterfactual_valid'] for r in failed),'records':records},'counterfactual_scope':'Memory-only WHEN diagnosis, not self-distillation output correction or admission.'}
OUT=ROOT/'docs/evidence/components/bootstrap_failure_audit_v1';OUT.mkdir(exist_ok=True)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='distill_multistep'}));print(json.dumps({k:v for k,v in result['distill_multistep'].items() if k!='records'}))
