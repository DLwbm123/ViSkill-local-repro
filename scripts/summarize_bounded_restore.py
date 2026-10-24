"""Operational two-input recovery gate, with all costs and failures retained."""
import os,json,time
from pathlib import Path
import numpy as np
root=Path(os.environ['TASK_ROOT']);name='bf16_delta_reload_v1';base=root/f'artifacts/component_tests/{name}'
p=json.loads((root/f'src/configs/smoke/{name}.json').read_text())
exit_info=json.loads((root/f'state/components/{name}_repair1_EXIT.json').read_text())
original_exit=json.loads((root/f'state/components/{name}_EXIT.json').read_text())
assert exit_info['exits']=={'full':0,'delta':0} and exit_info['reason'] is None
assert not exit_info['command_line_violations']
loaders={};cost={};packs={}
for phase in p['policy_order']:
    folder=base/p['policy_output_folders'][phase]
    cost[phase]=json.loads((folder/'COMPLETE.json').read_text())
    loaders[phase]=json.loads((folder/'actual_loading_info.json').read_text())
    assert cost[phase]['business_forward_samples']==cost[phase]['extra_alignment_forward_samples']==2
    assert not any(loaders[phase][k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    with np.load(folder/'scores.npz',allow_pickle=False) as data:packs[phase]={k:data[k] for k in data.files}
assert loaders['delta']['private_delta_applied'] and not loaders['full']['private_delta_applied']
a,b=packs['full'],packs['delta'];assert np.array_equal(a['response_mask'],b['response_mask']) and np.array_equal(a['seeds'],b['seeds'])
rows=[]
for i,seed in enumerate(a['seeds']):
    mask=a['response_mask'][i];d=np.abs(a['log_probs'][i,mask]-b['log_probs'][i,mask]);assert mask.any() and np.isfinite(d).all()
    maximum=float(d.max());mean=float(d.mean())
    rows.append({'seed':int(seed),'generated_tokens_scored':int(mask.sum()),'max_abs_logprob_difference_nats':maximum,'mean_abs_logprob_difference_nats':mean,'operational_gate_pass':maximum<=p['probability_gate']['per_row_max_abs_nats'] and mean<=p['probability_gate']['per_row_mean_abs_nats']})
result={'status':'COMPLETE_WITH_RETAINED_ZERO_SCORE_INITIALIZATION_FAILURE','completed_unix':time.time(),'rows':rows,'all_two_input_operational_gates_pass':all(r['operational_gate_pass'] for r in rows),'actual_loading_info':loaders,'cost':cost,'actual_actor_forward_samples':8,'generated_tokens':0,'environment_steps':0,'optimizer_steps':0,'new_model_snapshots':0,'original_cpu_configuration_assertion_failure_retained':True,'original_worker_initialization_failure':original_exit,'original_all_exit_zero_contract_pass':False,'exit':exit_info,
 'scope':'Existing Z inference artifact recovered over retained b0 and operationally compared on two fixed saved inputs only; no full-weight byte identity, FP32/optimizer resume, unseen-task generalization or learning-gain proof.'}
(base/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
