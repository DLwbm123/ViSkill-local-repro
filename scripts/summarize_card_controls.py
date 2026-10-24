"""Seal complete four-condition outcomes and saved-request diagnostics, no inference."""
import json,os,random,math,psutil
from pathlib import Path
from collections import Counter
from vagen.envs.frozenlake.utils.utils import parse_response
ROOT=Path(os.environ['TASK_ROOT']);NAME='card_content_controls_v1';OUT=ROOT/'artifacts/component_tests'/NAME
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());R=json.loads((OUT/'result.json').read_text());E=json.loads((ROOT/f'state/components/{NAME}_EXIT.json').read_text());receipt=json.loads((ROOT/f'state/components/{NAME}_receipt.json').read_text())
assert E['exits']=={'controls':0} and E['reason'] is None
assert len(R['records'])==64 and all(x['episodes']==16 for x in R['summary'].values())
seeds=P['frozen_seeds'];audit={};raw_count=0;initial={s:[] for s in seeds}
for arm in P['arms']:
 episodes=[(int(p.parent.name.split('_')[2]),json.loads(p.read_text())) for p in (OUT/arm/'episodes').glob('*/episode.json')];assert len(episodes)==16
 requests=[json.loads(p.read_text()) for p in (OUT/arm/'requests').glob('*.json')];raw_count+=len(requests);counts=Counter();positive=[]
 for q in requests:
  assert q['phase']=='dev' and q['policy_version']==0 and q['kind']=='rollout'
  parsed=parse_response(q['raw'],prompt_format='free_think',max_actions=5)
  counts['requests']+=1;counts['format_correct']+=bool(parsed['format_correct']);counts['unsupported_action_turns']+=any(a not in {'up','down','left','right'} for a in parsed['actions']);counts['stop_answers']+='<answer>Stop</answer>' in q['raw']
 for seed,ep in episodes:
  initial[seed].append(ep['initial_env_state']);assert not ep['distillation_audit'].get('model_calls',0)
  counts['holes']+=ep['terminal_reason']=='hole';counts['turn_limit']+=ep['terminal_reason']=='turn_limit'
  counts['boundary_noops']+=sum(a==b for a,b in zip(ep['trajectory_positions'],ep['trajectory_positions'][1:]))
  if ep['reward_support']['shadow_bonus_lambda_0_8']>0:
   q=[x for x in requests if x['seed']==seed]
   positive.append({'seed':seed,'shadow_bonus':ep['reward_support']['shadow_bonus_lambda_0_8'],'executed_actions':ep['trajectory_actions'],'stop_answers':sum('<answer>Stop</answer>' in x['raw'] for x in q),'turns':len(q)})
 audit[arm]={**dict(counts),'positive_shadow_failures':positive}
assert raw_count==R['model_calls']<=320
assert all(len(x)==4 and all(v==x[0] for v in x) for x in initial.values())
inputs=[json.loads(line) for line in (OUT/'input_shapes.jsonl').read_text().splitlines()];first={}
for q in inputs:first.setdefault((q['arm'],q['seed']),q)
assert len(first)==64
assert all(first['V',s]['prompt_tokens']==first['Blank',s]['prompt_tokens'] and first['V',s]['image_sizes']==first['Blank',s]['image_sizes'] for s in seeds)
byarm={a:{r['seed']:r for r in R['records'] if r['arm']==a} for a in P['arms']}
uncertainty={};rng=random.Random(20261024)
indices=[[rng.randrange(16) for _ in range(16)] for _ in range(10000)]
for pair in R['paired']:
 a,b=pair.split('-');uncertainty[pair]={}
 for metric in ['success','quality']:
  d=[int(byarm[a][s][metric])-int(byarm[b][s][metric]) for s in seeds]
  draws=sorted(sum(d[i] for i in ix)/16 for ix in indices)
  up=sum(x==1 for x in d);down=sum(x==-1 for x in d);n=up+down
  p_exact=min(1.0,2*sum(math.comb(n,k) for k in range(min(up,down)+1))/2**n) if n else 1.0
  uncertainty[pair][metric]={'difference':sum(d)/16,'descriptive_paired_map_bootstrap_95':[draws[249],draws[9749]],'posthoc_two_sided_exact_mcnemar_p':p_exact}
owned=[]
for q in psutil.process_iter(['pid','uids']):
 try:
  if q.info['uids'].real==os.getuid() and q.environ().get('TASK_RUN_TOKEN')==receipt['run_token']:owned.append(q.pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
review={'complete_episodes':64,'saved_requests':raw_count,'conditions_same_actual_initial_states':True,'V_Blank_first_prompt_lengths_and_image_dimensions_match_all16':True,'condition_audit':audit,'posthoc_uncertainty':uncertainty,'uncertainty_scope':'Descriptive within this selected16-map Up3 set, one skill and frozen policy; not natural-population or training-seed uncertainty. Not part of preregistered investment gate.','owned_residual_processes':owned,'baseline_bytes':receipt['baseline_bytes'],'peak_allocated_bytes':E['peak_allocated_bytes'],'peak_incremental_bytes':E['peak_allocated_bytes']-receipt['baseline_bytes'],'PPO_investment_gate':R['PPO_investment_gate']}
(OUT/'completion_review.json').write_text(json.dumps(review,indent=2));print(json.dumps({'summary':R['summary'],'paired':R['paired'],'PPO_investment_gate':R['PPO_investment_gate'],'audit':audit,'saved_requests':raw_count,'owned_residual_processes':owned,'peak_incremental_bytes':review['peak_incremental_bytes']}))
