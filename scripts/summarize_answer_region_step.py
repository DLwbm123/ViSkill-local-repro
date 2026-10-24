"""Full prospective decision for the single frozen loss-support intervention."""
import os,json,runpy,time
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);name='answer_region_step_v1';base=root/f'artifacts/component_tests/{name}'
p=json.loads((root/f'src/configs/smoke/{name}.json').read_text())
helper=runpy.run_path(str(root/'src/scripts/summarize_value_head_single_step.py'),run_name='paired_helpers');episodes,paired=helper['episodes'],helper['paired']
original_exit=json.loads((root/f'state/components/{name}_EXIT.json').read_text())
exit_info=json.loads((root/f'state/components/{name}_repair1_EXIT.json').read_text())
assert exit_info['exits']==dict.fromkeys(p['policy_order'],0) and exit_info['reason'] is None and not exit_info['command_line_violations']
train_folder=base/p['training_output_folder']
train=json.loads((train_folder/'COMPLETE.json').read_text());reload=json.loads((base/'reload/COMPLETE.json').read_text())
assert train['global_actor_optimizer_steps']==1 and reload['operational_reload_verified']
expected={r['seed'] for r in json.loads((base/'data_manifest.json').read_text())['splits']['dev']}
policies={};records=[];cost={};loaders={}
for policy in p['evaluation_order']:
    folder=base/policy;assert (folder/'COMPLETE.json').exists()
    rows=episodes(folder);assert len(rows)==32 and {r['seed'] for r in rows}==expected
    policies[policy]={r['seed']:r for r in rows};records.extend(dict(r,policy=policy) for r in rows)
    calls=int((folder/'model_calls.txt').read_text());assert calls==len(list((folder/'requests').glob('*.json')))<=160
    cost[policy]={'episodes':32,'native_calls':calls};loaders[policy]=json.loads((folder/'actual_loading_info.json').read_text())
    assert not any(loaders[policy][k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
groups={}
for group in ['all','conditional','natural_multi']:
    counts={};comparisons={}
    for policy,rows in policies.items():
        use={k:r for k,r in rows.items() if group=='all' or r['group']==group}
        counts[policy]={'episodes':len(use),'success':sum(r['success'] for r in use.values()),'quality':sum(r['quality'] for r in use.values()),'Stop_turns':sum(r['stop_turns'] for r in use.values()),'Stop_episodes':sum(r['has_stop'] for r in use.values()),'hole_episodes':sum(r['hole'] for r in use.values())}
    treatment={k:r for k,r in policies['A'].items() if group=='all' or r['group']==group}
    for control in ['b0','Z']:
        before={k:r for k,r in policies[control].items() if k in treatment};comparisons[control]={}
        for metric in ['success','quality','has_stop','hole']:
            result=paired(before,treatment,metric)
            if metric in ['has_stop','hole']:result['increased_cases']=result.pop('improved');result['decreased_cases']=result.pop('declined')
            comparisons[control][metric]=result
    groups[group]={'counts':counts,'A_vs':comparisons}
gate=p['primary_investment_gate'];conditional=groups['conditional'];natural=groups['natural_multi']
checks={'A_conditional_absolute_quality':conditional['counts']['A']['quality']>=gate['A_conditional_quality_at_least']}
for control in ['b0','Z']:
    for metric in ['success','quality']:
        comparison=conditional['A_vs'][control][metric]
        checks['conditional_'+metric+'_gain_vs_'+control]=comparison['net_count']>=gate['conditional_success_and_quality_gain_vs_each_b0_Z_at_least']
        checks['conditional_'+metric+'_paired_declines_vs_'+control]=comparison['declined']<=gate['conditional_paired_success_quality_declines_vs_each_at_most']
for metric in ['success','quality']:checks['natural_'+metric+'_vs_b0_non_decrease']=natural['A_vs']['b0'][metric]['net_count']>=0
for metric in ['Stop_turns','Stop_episodes','hole_episodes']:checks['natural_'+metric+'_vs_b0_non_increase']=natural['counts']['A'][metric]<=natural['counts']['b0'][metric]
assert sum(c['episodes'] for c in cost.values())==96 and sum(c['native_calls'] for c in cost.values())<=480
readiness=json.loads((root/'artifacts/component_tests/bf16_delta_reload_v1/result.json').read_text());assert readiness['actual_actor_forward_samples']==8
actual=train['business_forward_samples']+train['extra_alignment_forward_samples']+train['carried_business_forward_samples']+train['carried_extra_alignment_samples']+reload['business_forward_samples']+reload['extra_alignment_forward_samples'];assert actual==116
result={'status':'COMPLETE_WITH_RETAINED_PREUPDATE_METADATA_FAILURE','completed_unix':time.time(),'groups':groups,'gate_checks':checks,'prospective_investment_gate_pass':all(checks.values()),'records':records,'native_cost':cost,'train_cost':train,'reload_cost':reload,'actual_actor_forward_samples_before_native_evaluation':actual,'prior_artifact_readiness_actor_forward_samples':8,'method_total_actor_forward_samples_including_readiness':actual+8,'new_global_actor_steps':1,'critic_steps':0,'distill_calls':0,'admissions':0,'actual_loading_info':loaders,'inference_snapshot':json.loads((train_folder/'inference_snapshot.json').read_text()),'exit':exit_info,'original_exit':original_exit,'original_all_exit_zero_contract_pass':False,'original_actual_training_forwards_before_metadata_failure':0,'changed_factor':p['changed_factor'],
 'scope':'One seed and one step, fresh32 paired prospective pilot; all intervals/counts reported. Loss numerator intervention preserves full-generated denominator and full GAE/whitening. No proof of Stop causality, universal benefit, parameter-gradient attribution, original-paper reproduction or FP32/optimizer resume.'}
(base/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
