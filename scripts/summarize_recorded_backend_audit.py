"""Summarize the saved forward-only diagnostics; never invokes a model."""
import json,os,collections,re
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ['TASK_ROOT']);OUT=ROOT/'artifacts/component_tests/recorded_backend_audit_v2'

def main():
    score=json.loads((OUT/'score_results.json').read_text())
    audit=json.loads((OUT/'training_signal_audit.json').read_text())
    manifest=json.loads((OUT/'inputs.json').read_text())
    signals=torch.load(OUT/'first_batch_signals.pt',map_location='cpu',weights_only=True)
    role_values=collections.defaultdict(list)
    for case in score['cases']:
        pieces=case['token_pieces'];text=''.join(pieces)
        answer=re.search(r'<answer>(.*?)</answer>',text,re.S)
        thinking=re.search(r'<think>(.*?)</think>',text,re.S)
        cursor=0
        for piece,a,b in zip(pieces,case['actor_logprob'],case['native_logprob'],strict=True):
            end=cursor+len(piece)
            if piece in ['<|im_end|>','<|endoftext|>']:role='EOS'
            elif answer and end>answer.start(1) and cursor<answer.end(1):role='answer'
            elif thinking and end>thinking.start(1) and cursor<thinking.end(1):role='think'
            else:role='format_or_role'
            role_values[role].append(float(a-b));cursor=end
    role_summary={}
    for role,delta in role_values.items():
        arr=np.asarray(delta);ratio=np.exp(arr)
        role_summary[role]={'tokens':len(arr),'mean_abs_nats':float(np.abs(arr).mean()),
            'max_abs_nats':float(np.abs(arr).max()),'mean_signed_nats':float(arr.mean()),
            'ratio_min':float(ratio.min()),'ratio_max':float(ratio.max()),
            'tokens_ratio_outside_0_8_1_2':int(((ratio<.8)|(ratio>1.2)).sum())}
    groups=collections.defaultdict(list)
    for turn in audit['turns']:
        answer=turn['answer'].strip().lower()
        label='Stop' if answer=='stop' else 'other'
        groups[(turn['category'],label)].append(turn)
    grouped=[]
    for (category,label),turns in sorted(groups.items()):
        row={'category':category,'answer_group':label,'turns':len(turns),'immediate_reward_sum':sum(t['immediate_reward'] for t in turns)}
        for field,name in [('answer_advantages','original'),('answer_zero_baseline_advantages','zero_baseline'),('answer_per_turn_advantages','per_turn')]:
            values=[v for t in turns for v in t[field]]
            row[name]={'tokens':len(values),'positive_tokens':sum(v>0 for v in values),
                'mean':float(np.mean(values)) if values else None,'min':min(values) if values else None,'max':max(values) if values else None}
        row['turn_mean_sign_flips_original_to_zero']=sum(bool(np.mean(t['answer_advantages'])>0)!=bool(np.mean(t['answer_zero_baseline_advantages'])>0) for t in turns if t['answer_advantages'])
        row['turn_mean_sign_flips_original_to_per_turn']=sum(bool(np.mean(t['answer_advantages'])>0)!=bool(np.mean(t['answer_per_turn_advantages'])>0) for t in turns if t['answer_advantages'])
        grouped.append(row)
    mask=signals['response_mask'].bool()
    summary={'score_cases':[{k:v for k,v in c.items() if k not in ['actor_logprob','native_logprob','token_ids','token_pieces','ratio_actor_over_native']} for c in score['cases']],
        'all_eight_probability_gate_pass':score['all_eight_probability_gate_pass'],
        'strict_zero_generation_contract_pass':score['strict_zero_generation_contract_pass'],
        'native_returned_generated_tokens':score['native_returned_generated_tokens'],
        'score_token_roles':role_summary,
        'initial_head':json.loads((OUT/'initial_head_check.json').read_text()),
        'audit':{k:v for k,v in audit.items() if k!='turns'},'answer_groups':grouped,
        'original_values_std_generated':float(signals['values'][mask].std()),
        'original_advantages_mean_generated':float(signals['advantages'][mask].mean()),
        'cost':json.loads((OUT/'cost.json').read_text()),
        'training_seed_order':[manifest['episodes'][k]['seed'] for k in manifest['training_episode_ids']],
        'source_scope':'All comparisons are posthoc diagnostics of frozen saved requests/reconstructed images; no new training or causal probability-change test.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Numeric training signals are public research evidence, not model weights.
    arrays={k:(v.float() if v.dtype==torch.bfloat16 else v).numpy() for k,v in signals.items()}
    np.savez_compressed(OUT/'first_batch_signals.npz',**arrays)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
