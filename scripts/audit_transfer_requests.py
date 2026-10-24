"""Read-only original request parsing; no policy requests or environment actions."""
import json,os
from pathlib import Path
from collections import Counter
from vagen.envs.frozenlake.utils.utils import parse_response
ROOT=Path(os.environ['TASK_ROOT']);BASE=ROOT/'artifacts/component_tests/multi_card_transfer_v1'
result=json.loads((BASE/'result.json').read_text());summary={};positive=[]
for arm in ['empty','cards']:
 requests=[json.loads(p.read_text()) for p in sorted((BASE/arm/'requests').glob('*.json'))];counts=Counter()
 for d in requests:
  parsed=parse_response(d['raw'],prompt_format='free_think',max_actions=5)
  counts['requests']+=1;counts['format_correct']+=bool(parsed['format_correct']);counts['no_parsed_actions']+=not bool(parsed['actions']);counts['unsupported_action_turns']+=any(a not in {'up','down','left','right'} for a in parsed['actions']);counts['stop_answers']+='<answer>Stop</answer>' in d['raw']
 summary[arm]=dict(counts)
 if arm=='cards':
  for r in result['records']:
   if r['arm']=='cards' and r['reward_support']['shadow_bonus_lambda_0_8']>0:
    q=[d for d in requests if d['seed']==r['seed']]
    positive.append({'seed':r['seed'],'actions':r['actions'],'turns':r['turns'],'shadow_bonus':r['reward_support']['shadow_bonus_lambda_0_8'],'raw_turns':[d['raw'] for d in q],'parsed_actions':[parse_response(d['raw'],prompt_format='free_think',max_actions=5)['actions'] for d in q]})
old=ROOT/'results/ppo_smoke/bootstrap_card_pair_v1/B';regress=[]
for seed in [61001,61004,61042,61055,61058]:
 row={'seed':seed}
 for version in [0,4]:
  req=sorted([json.loads(p.read_text()) for p in (old/'requests').glob('*.json') if (d:=json.loads(p.read_text()))['seed']==seed and d['phase']=='dev' and d['policy_version']==version],key=lambda d:d['call'])
  row[str(version)]={'calls':len(req),'raw_turns':[d['raw'] for d in req],'parsed_actions':[parse_response(d['raw'],prompt_format='free_think',max_actions=5)['actions'] for d in req]}
 regress.append(row)
output={'model_calls':0,'transfer_request_parsing':summary,'positive_shadow_episodes':positive,'B_multi_regressions':regress,'scope':'Parsing actual saved requests does not verify original processed image tensors or logprob agreement.'}
(BASE/'request_audit.json').write_text(json.dumps(output,indent=2));print(json.dumps({'parsing':summary,'positive':[{k:v for k,v in d.items() if k!='raw_turns'} for d in positive]}))
