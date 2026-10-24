"""Saved-outcome/request failure audit; no inference, optimizer or library changes."""
import os,json,re,random,math
from pathlib import Path
from collections import Counter
from vagen.envs.frozenlake.utils.utils import parse_response
ROOT=Path(os.environ['TASK_ROOT']);NAME='fixed_library_ppo_v1';OUT=ROOT/'results/ppo_smoke'/NAME
result=json.loads((OUT/'result.json').read_text());records=result['records'];conditional={r['seed'] for r in records if r['fixed_group']=='conditional'}
training={};behavior={};selected_requests={}
for arm in ['A','B']:
 training[arm]={}
 for category in ['single_BFS1','multi_up3_applicable','medium_BFS2_3_without_multicard','long_BFS4plus']:
  rows=[r for r in records if r['arm']==arm and r['phase']=='train' and r['category']==category]
  training[arm][category]={'episodes':len(rows),'success':sum(r['traj_success'] for r in rows),'quality':sum(r['quality'] for r in rows)}
 reqs={}
 for p in (OUT/arm/'requests').glob('*.json'):
  d=json.loads(p.read_text())
  if d['phase']=='dev' and d.get('fixed_context')=='cards' and d['seed'] in conditional:
   reqs.setdefault((d['policy_version'],d['seed']),[]).append(d)
 selected_requests[arm]=reqs;behavior[arm]={}
 for version in [0,2,4]:
  rows=[r for r in records if r['arm']==arm and r['phase']=='dev' and r['fixed_context']=='cards' and r['fixed_group']=='conditional' and r['policy_version']==version];assert len(rows)==32
  counts=Counter();cases=[]
  for r in rows:
   q=sorted(reqs[version,r['seed']],key=lambda d:d['call']);parsed=[parse_response(x['raw'],prompt_format='free_think',max_actions=5) for x in q]
   counts['episodes']+=1;counts['retrieved_skill4']+=r['retrieved_skill_id']=='skill_4';counts['success']+=r['traj_success'];counts['quality']+=r['quality'];counts[r['terminal_reason']]+=1
   counts['requests']+=len(q);counts['tag_correct']+=sum(p['format_correct'] for p in parsed);counts['unsupported_action_turns']+=sum(any(a not in {'up','down','left','right'} for a in p['actions']) for p in parsed)
   stops=sum(p['actions']==['stop'] for p in parsed);counts['stop_only_turns']+=stops;counts['episodes_with_stop']+=stops>0
   pattern=r['trajectory_actions']==['up'] and stops==4 and len(q)==5;counts['up_then_4stop']+=pattern
   counts['at_512_output_token_limit']+=sum(len(d['output_ids'])>=512 for d in q)
   cases.append({'seed':r['seed'],'success':r['traj_success'],'actions':r['trajectory_actions'],'reason':r['terminal_reason'],'stop_turns':stops})
  behavior[arm][str(version)]={'counts':dict(counts),'cases':cases}
# Primary fixed-context differences, and natural multi-task paired changes.
comparisons={}
def subset(arm,version,group,multi=False):return {r['seed']:r for r in records if r['arm']==arm and r['phase']=='dev' and r['policy_version']==version and r['fixed_context']=='cards' and r['fixed_group']==group and (not multi or r['bfs']>1)}
before=subset('B',0,'conditional');after=subset('B',4,'conditional')
for name,t,c in [('B_self_conditional',after,before),('B_vs_A_conditional',after,subset('A',4,'conditional')),('B_self_natural_multi',subset('B',4,'natural',True),subset('B',0,'natural',True))]:
 comparisons[name]={};assert t.keys()==c.keys()
 for metric in ['traj_success','quality']:
  delta=[int(t[s][metric])-int(c[s][metric]) for s in sorted(t)];n=len(delta);rng=random.Random(20261024);draws=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(10000));up=sum(d==1 for d in delta);down=sum(d==-1 for d in delta);discord=up+down
  comparisons[name][metric]={'pairs':n,'improved':up,'worsened':down,'net':up-down,'difference':sum(delta)/n,'posthoc_descriptive_paired_map_95':[draws[249],draws[9749]],'posthoc_exact_mcnemar_p':min(1.,2*sum(math.comb(discord,k) for k in range(min(up,down)+1))/2**discord) if discord else 1.}
regressions=[]
for seed,r in before.items():
 if r['traj_success'] and not after[seed]['traj_success']:
  regressions.append({'seed':seed,'before_actions':r['trajectory_actions'],'after_actions':after[seed]['trajectory_actions'],'after_reason':after[seed]['terminal_reason'],'before_raw':[x['raw'] for x in sorted(selected_requests['B'][0,seed],key=lambda x:x['call'])],'after_raw':[x['raw'] for x in sorted(selected_requests['B'][4,seed],key=lambda x:x['call'])]})
metrics={}
keys=['actor/ppo_kl','actor/pg_clipfrac','actor/entropy','critic/vf_loss','critic/vf_clipfrac','critic/vpred_mean','critic/returns/mean','critic/score/mean']
for arm in ['A','B']:
 extracted=[]
 for line in (ROOT/f'logs/ppo_smoke/{NAME}_{arm}.log').read_text(errors='replace').splitlines():
  if 'actor/pg_loss:' not in line:continue
  step=re.search(r'step:\s*(\d+)',line);row={'step':int(step.group(1)) if step else None}
  for k in keys:
   m=re.search(re.escape(k)+r':\s*([-+0-9.eE]+)',line)
   if m:row[k]=float(m.group(1))
  extracted.append(row)
 metrics[arm]=extracted
v={'model_calls':0,'training':training,'conditional_cards_behavior':behavior,'paired':comparisons,'B_lost_initial_successes':regressions,'available_console_metrics':metrics,'causal_scope':'Observed behavioral failure and logged numerical summaries; no token-level advantage attribution or causal diagnosis from these aggregates alone. No reruns or reward edits.'}
(OUT/'failure_audit.json').write_text(json.dumps(v,indent=2));print(json.dumps({'training':training,'behavior_counts':{a:{v:d['counts'] for v,d in b.items()} for a,b in behavior.items()},'paired':comparisons,'regressions':len(regressions),'metric_steps':{a:[x['step'] for x in q] for a,q in metrics.items()}}))
