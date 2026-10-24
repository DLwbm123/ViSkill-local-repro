"""Complete paired prospective panel; no model calls or test access."""
import os,json,runpy,time
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);NAME='prospective_value_head_eval_v1';BASE=ROOT/f'artifacts/component_tests/{NAME}'
helpers=runpy.run_path(str(ROOT/'src/scripts/summarize_value_head_single_step.py'),run_name='paired_helpers')
episodes,paired=helpers['episodes'],helpers['paired']

def main():
    p=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
    manifest=json.loads((BASE/'data_manifest.json').read_text());expected={r['seed'] for r in manifest['splits']['dev']}
    policies={};records=[];cost={};loading={}
    for name in p['policy_order']:
        folder=BASE/p['policy_output_folders'][name];assert (folder/'COMPLETE.json').exists(),'partial policy evaluation stays partial'
        rows=episodes(folder);assert len(rows)==32 and {r['seed'] for r in rows}==expected
        policies[name]={r['seed']:r for r in rows};records.extend(dict(r,policy=name) for r in rows)
        calls=int((folder/'model_calls.txt').read_text());assert calls==len(list((folder/'requests').glob('*.json')))<=160
        cost[name]={'episodes':32,'native_calls':calls}
        loading[name]=json.loads((folder/'actual_loading_info.json').read_text())
        assert not loading[name]['unexpected_keys'] and not loading[name]['mismatched_keys'] and not loading[name]['error_msgs']
    groups={}
    for group in ['all','conditional','natural_multi']:
        groups[group]={'counts':{},'Z_vs':{}}
        for name,rows in policies.items():
            use={k:r for k,r in rows.items() if group=='all' or r['group']==group}
            groups[group]['counts'][name]={'episodes':len(use),'success':sum(r['success'] for r in use.values()),'quality':sum(r['quality'] for r in use.values()),'Stop_episodes':sum(r['has_stop'] for r in use.values()),'Stop_turns':sum(r['stop_turns'] for r in use.values()),'hole_episodes':sum(r['hole'] for r in use.values())}
        z={k:r for k,r in policies['Z'].items() if group=='all' or r['group']==group}
        for control in ['b0','O']:
            base={k:r for k,r in policies[control].items() if k in z};comparison={}
            for metric in ['success','quality','has_stop','hole']:
                result=paired(base,z,metric)
                if metric in ['has_stop','hole']:
                    result['increased_cases']=result.pop('improved');result['decreased_cases']=result.pop('declined')
                comparison[metric]=result
            groups[group]['Z_vs'][control]=comparison
    conditional=groups['conditional'];natural=groups['natural_multi']
    checks={'Z_conditional_quality_absolute_at_least4':conditional['counts']['Z']['quality']>=4}
    for name in ['b0','O']:
        q=conditional['Z_vs'][name]
        checks['Z_conditional_quality_gain_vs_'+name+'_at_least2']=q['quality']['net_count']>=2
        for metric in ['success','quality']:checks['Z_conditional_'+metric+'_declines_vs_'+name+'_at_most1']=q[metric]['declined']<=1
    for metric in ['success','quality']:
        q=natural['Z_vs']['b0'][metric]
        checks['natural_'+metric+'_non_decrease']=q['net_count']>=0
        checks['natural_'+metric+'_zero_paired_declines']=q['declined']==0
    checks['natural_Stop_turns_non_increase']=natural['counts']['Z']['Stop_turns']<=natural['counts']['b0']['Stop_turns']
    checks['natural_Stop_episodes_non_increase']=natural['Z_vs']['b0']['has_stop']['net_count']<=0
    checks['natural_holes_non_increase']=natural['Z_vs']['b0']['hole']['net_count']<=0
    assert sum(v['episodes'] for v in cost.values())==96 and sum(v['native_calls'] for v in cost.values())<=480
    result={'status':'COMPLETE_WITH_RETAINED_ZERO_QUERY_INITIALIZATION_FAILURE','original_zero_failed_attempts_pass':False,'failed_initialization_business_queries':0,'inference_only_snapshot_reload_verified':True,'full_FP32_optimizer_resume_verified':False,'completed_unix':time.time(),'groups':groups,'gate_checks':checks,'prospective_investment_gate_pass':all(checks.values()),
        'cost':cost,'optimizer_steps':0,'distill_calls':0,'admissions':0,'new_snapshots':0,'actual_loading_info':loading,
        'scientific_generalization_or_original_collapse_cause_established':False,
        'records':records,'interval_method':'Same conservative finite-sample paired95% Clopper-Pearson/Bonferroni difference bounds as report20, full denominators and exact discordant sign p; no multiplicity-adjusted efficacy claim.',
        'limitations':['One frozen training batch/one step per trained policy; no training-seed replication','BF16 inference snapshots, not FP32/optimizer restoration','Serial scheduling deliberately differs from previous batched Dev and is common to all three policies','Original cross-backend think-token probability gate remains failed','Historical Stop mechanism gate remains failed and is not rescued by performance']}
    (BASE/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
