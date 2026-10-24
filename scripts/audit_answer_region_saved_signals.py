"""Posthoc saved-logprob description; no model, gradient or new sample."""
import os,json,time,signal
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
root=Path(os.environ['TASK_ROOT']);base=root/'artifacts/component_tests/answer_region_step_v1'
plan=json.loads((base/'posthoc_signal_plan.json').read_text())
deadline=min(plan['preparation_started_unix']+plan['max_seconds_including_preparation'],1792884169.007)
def stop(*args):raise TimeoutError('original posthoc preparation-inclusive deadline')
signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(deadline-time.time())))
assert time.time()<deadline and not (base/'posthoc_signal_audit.json').exists()
old=root/'artifacts/component_tests/value_head_single_step_v1'
rows=json.loads((old/'data_manifest.json').read_text())['splits']['train']
collection={r['seed']:r for r in json.loads((old/'C/collection_gate.json').read_text())['rows']}
with np.load(old/'Z/training_signals.npz',allow_pickle=False) as data:z={k:data[k] for k in ['old_log_probs','post_log_probs','advantages','response_mask']}
with np.load(base/'train_repair1/training_signals.npz',allow_pickle=False) as data:a={k:data[k] for k in data.files}
with np.load(root/'artifacts/component_tests/actor_loss_support_audit_v1/loss_support.npz',allow_pickle=False) as data:roles=data['role_ids'].copy()
full=z['response_mask'].astype(bool);answer=roles==1
assert len(rows)==32 and np.array_equal(full,a['response_mask'].astype(bool)) and np.array_equal(roles>0,full)
assert np.array_equal(z['advantages'],a['advantages_before'])
assert np.array_equal(a['advantages_after'],z['advantages']*answer)
names={1:'answer_content_overlap',2:'think_content_overlap',3:'EOS',4:'format_role_or_unparsed'}
groups={f'{r["category"]}_{"success" if collection[r["seed"]]["success"] else "failure"}' for r in rows}
branches={'Z':(z['post_log_probs']-z['old_log_probs']),'A':(a['post_log_probs']-a['pre_log_probs'])}
assert all(np.isfinite(delta[full]).all() for delta in branches.values())
stats={};per_episode=[]
for group in ['all']+sorted(groups):
    selected=np.array([group=='all' or group==f'{r["category"]}_{"success" if collection[r["seed"]]["success"] else "failure"}' for r in rows])
    stats[group]={'episodes':int(selected.sum()),'roles':{}}
    for number,name in names.items():
        mask=(roles==number)&selected[:,None];summary={'tokens':int(mask.sum()),'episodes_with_tokens':int(mask.any(-1).sum()),'branches':{}}
        for branch,delta in branches.items():
            values=delta[mask];means=[float(delta[i,m].mean()) for i,m in enumerate(mask) if m.any()]
            adv=z['advantages'][mask];nonzero=adv!=0
            summary['branches'][branch]={'token_weighted_mean_delta_nats':float(values.mean()) if len(values) else None,'median_delta_nats':float(np.median(values)) if len(values) else None,'episode_weighted_mean_delta_nats':float(np.mean(means)) if means else None,'positive_delta_tokens':int((values>0).sum()),'negative_delta_tokens':int((values<0).sum()),'zero_delta_tokens':int((values==0).sum()),'positive_original_advantage_tokens':int((adv>0).sum()),'original_advantage_and_delta_same_sign_tokens':int(((values*adv)>0).sum()),'nonzero_original_advantage_tokens':int(nonzero.sum())}
        stats[group]['roles'][name]=summary
for i,row in enumerate(rows):
    item={'seed':row['seed'],'category':row['category'],'collection_success':collection[row['seed']]['success'],'roles':{}}
    for number,name in names.items():
        mask=roles[i]==number
        item['roles'][name]={'tokens':int(mask.sum()),'Z_mean_delta_nats':float(branches['Z'][i,mask].mean()) if mask.any() else None,'A_mean_delta_nats':float(branches['A'][i,mask].mean()) if mask.any() else None}
    per_episode.append(item)
pre_diff=np.abs(a['pre_log_probs']-z['old_log_probs'])[full]
result={'status':'COMPLETE_POSTHOC_CPU_ONLY','completed_unix':time.time(),'all32_rows_preserved':True,'generated_tokens':int(full.sum()),'answer_overlap_tokens':int(answer.sum()),'pre_A_carried_vs_Z_max_abs_nats':float(pre_diff.max()),'full_advantages_before_mask_exact':True,'nonanswer_advantages_after_mask_exactly_zero':True,'strata':stats,'per_episode':per_episode,'model_forwards':0,'model_backwards':0,'optimizer_steps':0,'environment_steps':0,'prospective_gate_remains_failed':True,'limitations':['Posthoc plan fixed after all prospective results were known; no new heldout validation','Token/episode summaries are descriptive and no token-level significance tests are used','Shared parameter updates can change zero-direct-coefficient regions; no gradient-norm or Adam attribution','Carried A pre scores and post scores are from different same-source worker instances after preupdate metadata repair','Training-logprob patterns do not establish the cause of natural-panel decline or Stop behavior','No new mask/LR/seed search, model call or actor step is authorized by this audit']}
encoded=json.dumps(result,indent=2).encode();assert len(encoded)<plan['maximum_additional_bytes'] and time.time()<deadline
(base/'posthoc_signal_audit.json').write_bytes(encoded);signal.alarm(0)
print(json.dumps({'status':result['status'],'bytes':len(encoded),'pre_max_difference':result['pre_A_carried_vs_Z_max_abs_nats'],'all_roles':stats['all']}))
