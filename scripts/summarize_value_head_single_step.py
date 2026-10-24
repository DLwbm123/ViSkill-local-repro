"""Read all frozen single-step evidence; no generation or model forward."""
import os,json,re,time,collections
from pathlib import Path
import numpy as np
from scipy.stats import beta,binomtest

ROOT=Path(os.environ['TASK_ROOT']);BASE=ROOT/'artifacts/component_tests/value_head_single_step_v1'

def interval(wins,losses,n,alpha=.05):
    # Two simultaneous binomial CP intervals bound the paired multinomial gap.
    def cp(k):
        return (0. if k==0 else float(beta.ppf(alpha/4,k,n-k+1)),
                1. if k==n else float(beta.ppf(1-alpha/4,k+1,n-k)))
    a,b=cp(wins);c,d=cp(losses)
    return [max(-1.,a-d),min(1.,b-c)]

def paired(before,after,metric):
    assert before.keys()==after.keys()
    wins=sum(after[k][metric] and not before[k][metric] for k in before)
    losses=sum(before[k][metric] and not after[k][metric] for k in before)
    n=len(before);discordant=wins+losses
    return {'n':n,'before':sum(before[k][metric] for k in before),'after':sum(after[k][metric] for k in after),
        'improved':wins,'declined':losses,'net_count':wins-losses,'paired_rate_change':(wins-losses)/n,
        'conservative_exact_95_interval':interval(wins,losses,n),
        'exact_discordant_sign_p_two_sided':float(binomtest(wins,discordant,.5).pvalue) if discordant else 1.}

def episodes(folder):
    from vagen.envs.frozenlake.utils.utils import parse_response
    requests=collections.defaultdict(list)
    for path in (folder/'requests').glob('*.json'):
        r=json.loads(path.read_text())
        if r['kind']=='rollout':requests[r['request_id']].append(r)
    rows=[]
    for path in (folder/'episodes').glob('*/episode.json'):
        ep=json.loads(path.read_text());raw=requests[ep['mechanism_audit']['request_id']]
        assert len(raw)==len([s for s in ep['mechanism_audit']['spans'] if s['kind']=='generated'])
        actions=[parse_response(r['raw'],max_actions=5)['actions'] for r in raw]
        stops=sum('stop' in a for a in actions)
        rows.append({'seed':ep['seed'],'phase':ep['phase'],'version':ep['policy_version'],'group':ep['fixed_group'],
            'category':ep['category'],'bfs':ep['bfs'],'success':bool(ep['traj_success']),
            'quality':bool(ep['traj_success'] and len(ep['trajectory_actions'])<=ep['bfs']),
            'actions':ep['trajectory_actions'],'action_count':len(ep['trajectory_actions']),'turns':len(raw),
            'stop_turns':stops,'has_stop':stops>0,'hole':ep['terminal_reason']=='hole','terminal_reason':ep['terminal_reason'],
            'retrieved_skill_id':ep['retrieved_skill_id'],'episode_path':str(path)})
    return rows

def fresh_signals(folder,rows,tokenizer):
    with np.load(folder/'training_signals.npz',allow_pickle=False) as pack:
        arrays={key:pack[key] for key in pack.files}
    frozen=json.loads((BASE/'data_manifest.json').read_text())['splits']['train']
    selected={r['seed']:r for r in rows};results=[]
    for i,case in enumerate(frozen):
        ep=json.loads(Path(selected[case['seed']]['episode_path']).read_text())
        response=arrays['input_ids'][i,-arrays['response_mask'].shape[1]:]
        for turn,span in enumerate(s for s in ep['mechanism_audit']['spans'] if s['kind']=='generated'):
            begin,end=span['start'],span['end'];ids=response[begin:end].tolist()
            text=tokenizer.decode(ids,skip_special_tokens=True);answer=re.search(r'<answer>(.*?)</answer>',text,re.S)
            if not answer:positions=[];answer_text=''
            else:
                prefix=[len(tokenizer.decode(ids[:k],skip_special_tokens=True)) for k in range(len(ids)+1)]
                positions=[begin+k for k in range(len(ids)) if prefix[k+1]>answer.start(1) and prefix[k]<answer.end(1)]
                answer_text=answer.group(1).strip()
            old=arrays['old_log_probs'][i,positions];post=arrays['post_log_probs'][i,positions]
            results.append({'seed':case['seed'],'category':case['category'],'success':bool(ep['traj_success']),'turn':turn+1,
                'answer':answer_text,'answer_overlapping_token_positions':positions,'answer_token_ids':[int(response[p]) for p in positions],
                'old_log_probs':old.tolist(),'post_log_probs':post.tolist(),'delta_log_probs':(post-old).tolist(),
                'advantages':arrays['advantages'][i,positions].tolist(),
                'scope':'Tokens overlapping answer content; may include punctuation or boundary text, no invented action-only logits.'})
    strata={}
    for category in ['target_success','target_failure','other_success','other_failure']:
        for stop in [False,True]:
            use=[r for r in results if ('target' if r['category']=='multi_up3_applicable' else 'other')+'_'+('success' if r['success'] else 'failure')==category and (r['answer'].lower()=='stop')==stop]
            vals=[v for r in use for v in r['delta_log_probs']];adv=[v for r in use for v in r['advantages']]
            strata[category+('_Stop' if stop else '_nonStop')]={'turns':len(use),'tokens':len(vals),'mean_delta_logprob':float(np.mean(vals)) if vals else None,'mean_advantage':float(np.mean(adv)) if adv else None,'positive_delta_tokens':sum(v>0 for v in vals),'positive_advantage_tokens':sum(v>0 for v in adv)}
    (folder/'fresh_answer_signals.json').write_text(json.dumps({'turns':results,'strata':strata},indent=2))
    return strata,arrays

def main():
    if os.getenv('SUMMARY_SELF_CHECK')=='1':
        a={i:{'x':False} for i in range(8)};b={i:{'x':True} for i in range(8)}
        assert paired(a,b,'x')['net_count']==8 and paired(a,a,'x')['net_count']==0
        assert interval(0,0,8)[0]<0<interval(0,0,8)[1]
        assert interval(8,0,8)[0]>0 and interval(0,8,8)[1]<0
        print('paired exact interval and concordant/discordant cases PASS');return
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    folders={'O':BASE/'O_repair1','Z':BASE/'Z'}
    for folder in folders.values():assert (folder/'COMPLETE.json').exists(),'keep partial result partial'
    manifest=json.loads((BASE/'data_manifest.json').read_text());expected={r['seed'] for r in manifest['splits']['dev']}
    train=episodes(BASE/'C');allrows=[];branches={};arrays={}
    old=json.loads((ROOT/'artifacts/component_tests/recorded_backend_audit_v2/inputs.json').read_text())
    cfg=OmegaConf.load(old['original_resolved_config']);tokenizer=AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path,trust_remote_code=True)
    for name,folder in folders.items():
        rows=episodes(folder)
        if name=='O':rows=episodes(BASE/'O')+rows
        assert len(rows)==32
        before={r['seed']:r for r in rows if r['version']==0};after={r['seed']:r for r in rows if r['version']==1}
        assert before.keys()==after.keys()==expected
        branch={'groups':{},'baseline_carried':name=='O','completion':json.loads((folder/'COMPLETE.json').read_text()),'actor_metrics':json.loads((folder/'actor_metrics.json').read_text()),'inference_snapshot':json.loads((folder/'inference_snapshot.json').read_text())}
        for group in ['all','conditional','natural_multi']:
            x={k:r for k,r in before.items() if group=='all' or r['group']==group};y={k:after[k] for k in x}
            branch['groups'][group]={'success':paired(x,y,'success'),'quality':paired(x,y,'quality'),
                'stop_episodes':paired(x,y,'has_stop'),'hole_episodes':paired(x,y,'hole'),
                'before_stop_turns':sum(r['stop_turns'] for r in x.values()),'after_stop_turns':sum(r['stop_turns'] for r in y.values())}
        branch['fresh_answer_strata'],arrays[name]=fresh_signals(folder,train,tokenizer)
        branches[name]=branch
        allrows.extend(dict(r,branch=name) for r in rows)
    mask=arrays['O']['response_mask'].astype(bool)
    assert np.array_equal(mask,arrays['Z']['response_mask'].astype(bool))
    assert np.array_equal(arrays['O']['input_ids'],arrays['Z']['input_ids'])
    difference=np.abs(arrays['O']['old_log_probs']-arrays['Z']['old_log_probs'])[mask]
    alignment={'max_abs_nats':float(difference.max()),'mean_abs_nats':float(difference.mean()),'gate_pass':bool(difference.max()<=.05 and difference.mean()<=.005)}
    probes={name:json.loads((folder/'stop_probe_result.json').read_text())['rows'] for name,folder in folders.items()}
    contrasts=[]
    for a,b in zip(probes['O'],probes['Z'],strict=True):
        assert a['seed']==b['seed'] and a['turn']==b['turn']
        contrasts.append({'seed':a['seed'],'turn':a['turn'],'O_pre':a['pre_log_prob'],'O_post':a['post_log_prob'],'O_delta':a['delta_log_prob'],'Z_pre':b['pre_log_prob'],'Z_post':b['post_log_prob'],'Z_delta':b['delta_log_prob'],'Z_minus_O_delta':b['delta_log_prob']-a['delta_log_prob']})
    d=[r['Z_minus_O_delta'] for r in contrasts]
    mechanism=alignment['gate_pass'] and all(v<=0 for v in d) and float(np.mean(d))<0
    z=branches['Z']['groups'];o=branches['O']['groups']
    checks={'Z_own_conditional_quality_gain_at_least1':z['conditional']['quality']['net_count']>=1,
        'Z_conditional_quality_gain_over_O_at_least1':z['conditional']['quality']['net_count']-o['conditional']['quality']['net_count']>=1,
        'Z_natural_success_non_decrease':z['natural_multi']['success']['net_count']>=0,
        'Z_natural_quality_non_decrease':z['natural_multi']['quality']['net_count']>=0,
        'Z_natural_Stop_turns_non_increase':z['natural_multi']['after_stop_turns']<=z['natural_multi']['before_stop_turns'],
        'Z_natural_Stop_episodes_non_increase':z['natural_multi']['stop_episodes']['net_count']<=0,
        'Z_natural_hole_episodes_non_increase':z['natural_multi']['hole_episodes']['net_count']<=0}
    costs={}
    for phase in ['C','O','O_repair1','Z']:
        folder=BASE/phase;count=int((folder/'model_calls.txt').read_text())
        assert count==len(list((folder/'requests').glob('*.json')))
        costs[phase]={'native_calls':count,'actual_episodes':len(list((folder/'episodes').glob('*/episode.json')))}
    assert sum(v['native_calls'] for v in costs.values())<=512 and sum(v['actual_episodes'] for v in costs.values())==96
    result={'status':'COMPLETE_WITH_RETAINED_PREUPDATE_FAILURE','completed_unix':time.time(),'collection_gate':json.loads((BASE/'C/collection_gate.json').read_text()),
        'branches':branches,'paired_pre_actor_alignment':alignment,'historical_stop_probe_contrasts':contrasts,
        'localized_mechanism_gate_pass':bool(mechanism),'evaluation_gate_checks':checks,'evaluation_followup_gate_pass':all(checks.values()),
        'costs':costs,'actual_actor_forward_samples_including_failed_attempt':280,'actual_critic_forward_samples_including_failed_attempt':96,
        'actual_global_actor_steps':2,'actual_critic_steps':0,'admissions':0,'training_Stop_turns':sum(r['stop_turns'] for r in train),'original_no_repeat_scoring_plan_pass':False,
        'stable_generalization_or_original_collapse_cause_established':False,
        'interval_method':'Conservative finite-sample paired rate-difference95% interval from two simultaneous97.5% Clopper-Pearson bounds on discordant win/loss probabilities; Bonferroni valid without within-pair independence. Exact two-sided sign p on discordant pairs also reported. Small fixed developed panels, no multiplicity-adjusted efficacy claim.',
        'records':allrows,'training_records':train,
        'limitations':['Known native/actor think-token discrepancy remains failed','O baseline carried from its original b0 instance after serialization failure; both instances load same retained b0 and empty optimizer, no full-weight identity claim','Two historical Stop probes were selected posthoc from the previous published audit; fresh32 have zero parsed Stop turns, so this is transfer rather than a direct same-Stop-token gradient experiment','Inference snapshots retained with readable headers, not independently reloaded; no optimizer/full training recovery']}
    (BASE/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','training_records','branches','collection_gate']},indent=2))

if __name__=='__main__':main()
