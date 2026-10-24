"""Close the frozen pair from saved records; no model dispatch or resampling."""
import json,os,runpy,time
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);NAME='support_matched_curriculum_v1';BASE=ROOT/f'artifacts/component_tests/{NAME}'
def read(path):return json.loads(path.read_text())
def main():
    p=read(ROOT/f'src/configs/smoke/{NAME}.json');end=read(ROOT/f'state/components/{NAME}_EXIT.json')
    helper=runpy.run_path(str(ROOT/'src/scripts/summarize_value_head_single_step.py'),run_name='paired_helpers');episodes,paired=helper['episodes'],helper['paired']
    completed={phase:read(BASE/phase/'COMPLETE.json') for phase in p['phase_order'] if (BASE/phase/'COMPLETE.json').exists()}
    bank=read(BASE/'collect/bank_ready.json') if (BASE/'collect/bank_ready.json').exists() else None
    native={phase:{'episodes':len(list((BASE/phase/'episodes').glob('*/episode.json'))),'native_calls':int((BASE/phase/'model_calls.txt').read_text()) if (BASE/phase/'model_calls.txt').exists() else 0} for phase in p['phase_order'] if phase=='collect' or phase.startswith('eval_')}
    audits={label:[json.loads(line) for f in (BASE/label/'update_audit').glob('actor_*.jsonl') for line in f.read_text().splitlines()] for label in ['Q','R']}
    result={'completed_unix':time.time(),'status':'CLOSED_WITH_ENGINEERING_FAILURE','exit':end,'phase_completions':completed,'native_cost':native,'bank_ready':bank,'actual_optimizer_audit_records':audits,'new_global_actor_steps':0 if not any(audits.values()) and not any((BASE/label/'update_started.json').exists() for label in ['Q','R']) else None,'groups':{},'gate_checks':{},'prospective_investment_gate_pass':None,'critic_steps':0,'distill_calls':sum(json.loads(f.read_text())['kind']=='distill' for phase in p['phase_order'] for f in (BASE/phase/'requests').glob('*.json')),'admissions':None,'scope':'One frozen data-composition intervention, one training seed and one step per arm. Wide paired intervals; no stable efficacy claim. Private bank, deltas and schemas excluded from public evidence.'}
    if bank is not None and not bank['gate_pass']:
        assert set(completed)=={'collect'} and not any(audits.values())
        result['status']='COMPLETE_COLLECTION_FEASIBILITY_GATE_FAILED';result['new_global_actor_steps']=0
    elif end['reason'] is None and end['exits']==dict.fromkeys(p['phase_order'],0):
        assert set(completed)==set(p['phase_order']) and not end['command_line_violations']
        assert bank['gate_pass'] and result['distill_calls']==0 and all(len(audits[label])==2 and {r['rank'] for r in audits[label]}=={0,1} and all(r['finite'] and r['sample_delta_l2']>0 for r in audits[label]) for label in ['Q','R'])
        assert all(completed[f'reload_{label}']['operational_reload_verified'] and completed[label]['global_actor_steps']==1 for label in ['Q','R'])
        expected={r['seed']:r for r in read(BASE/'data_manifest.json')['splits']['dev']};policies={};records=[]
        for label in p['evaluation_labels']:
            folder=BASE/f'eval_{label}';rows=episodes(folder);assert len(rows)==40 and {r['seed'] for r in rows}==expected.keys()
            for row in rows:
                assert row['category']==expected[row['seed']]['category'] and row['group']==expected[row['seed']]['group']
                row.pop('episode_path',None)
            policies[label]={r['seed']:r for r in rows};records.extend(dict(r,policy=label) for r in rows)
            info=read(folder/'actual_loading_info.json');assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
            assert native[f'eval_{label}']['native_calls']==len(list((folder/'requests').glob('*.json')))<=200
        groups={}
        for group in ['all','conditional','natural_multi','medium','long','single']:
            selected={seed for seed,r in expected.items() if group=='all' or r['group']==group or r['category']==group}
            counts={};comparisons={}
            for label,rows in policies.items():
                use=[r for seed,r in rows.items() if seed in selected]
                counts[label]={'episodes':len(use),'success':sum(r['success'] for r in use),'quality':sum(r['quality'] for r in use),'Stop_turns':sum(r['stop_turns'] for r in use),'Stop_episodes':sum(r['has_stop'] for r in use),'hole_episodes':sum(r['hole'] for r in use)}
            treatment={seed:policies['R'][seed] for seed in selected}
            for control in ['b0','Q']:
                before={seed:policies[control][seed] for seed in selected};comparisons[control]={}
                for metric in ['success','quality','has_stop','hole']:
                    value=paired(before,treatment,metric)
                    if metric in ['has_stop','hole']:value['increased_cases']=value.pop('improved');value['decreased_cases']=value.pop('declined')
                    comparisons[control][metric]=value
            groups[group]={'counts':counts,'R_vs':comparisons}
        g=p['investment_gate'];checks={}
        for group,key in [('conditional','R_conditional_quality_minimum'),('long','R_long_quality_minimum'),('single','R_single_quality_minimum')]:checks[group+'_absolute_quality']=groups[group]['counts']['R']['quality']>=g[key]
        for metric in ['success','quality']:
            checks['natural_absolute_'+metric]=groups['natural_multi']['counts']['R'][metric]>=g['R_natural_success_quality_minimum']
            for control in ['b0','Q']:
                for group,gain_key,decline_key in [('conditional','R_conditional_success_quality_gain_vs_b0_minimum' if control=='b0' else 'R_conditional_success_quality_net_vs_Q_minimum','conditional_paired_declines_vs_both_maximum'),('natural_multi','R_natural_success_quality_gain_vs_both_minimum','natural_paired_declines_vs_both_maximum'),('single','single_success_quality_net_vs_both_minimum','single_success_quality_paired_declines_vs_both_maximum')]:
                    comparison=groups[group]['R_vs'][control][metric];checks[f'{group}_{metric}_gain_vs_{control}']=comparison['net_count']>=g[gain_key];checks[f'{group}_{metric}_declines_vs_{control}']=comparison['declined']<=g[decline_key]
                checks[f'long_{metric}_gain_vs_{control}']=groups['long']['R_vs'][control][metric]['net_count']>=g['R_long_success_quality_gain_vs_both_minimum']
        for group in ['natural_multi','single']:
            for control in ['b0','Q']:
                for metric in ['Stop_turns','Stop_episodes','hole_episodes']:checks[f'{group}_{metric}_vs_{control}']=groups[group]['counts']['R'][metric]<=groups[group]['counts'][control][metric]
        actual=sum(completed[label]['business_forward_samples']+completed[label]['extra_alignment_samples']+completed[f'reload_{label}']['business_forward_samples']+completed[f'reload_{label}']['extra_alignment_samples'] for label in ['Q','R']);assert actual==232
        result.update(status='COMPLETE_PAIRED_CURRICULUM_PILOT',groups=groups,records=records,gate_checks=checks,prospective_investment_gate_pass=all(checks.values()),new_global_actor_steps=2,admissions=0,actual_actor_forward_samples_before_evaluation=actual)
    if (BASE/'failure_audit.json').exists():
        audit=read(BASE/'failure_audit.json');result['failure_audit']=audit;result['admissions']=audit['admissions']
    assert sum(r['native_calls'] for r in native.values())<=p['limits']['total_native_calls']
    (BASE/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['records','actual_optimizer_audit_records']},indent=2))
if __name__=='__main__':main()
