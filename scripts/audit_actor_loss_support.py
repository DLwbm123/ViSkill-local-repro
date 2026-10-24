"""Posthoc scalar logprob-loss support, not parameter-gradient attribution."""
import os,json,time,re
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

ROOT=Path(os.environ['TASK_ROOT']);NAME='actor_loss_support_audit_v1'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text())
OUT=ROOT/f'artifacts/component_tests/{NAME}'
assert not OUT.exists(),'refuse replacing this posthoc audit'
OUT.mkdir(parents=True)
assert time.time()<P['preparation_started_unix']+P['max_seconds_including_preparation']
BASE=ROOT/'artifacts/component_tests/value_head_single_step_v1'
manifest=json.loads((BASE/'data_manifest.json').read_text());rows=manifest['splits']['train']
episodes={}
for path in (BASE/'C/episodes').glob('*/episode.json'):
    episode=json.loads(path.read_text());episodes[episode['seed']]=episode
assert len(rows)==len(episodes)==32
old=json.loads((ROOT/'artifacts/component_tests/recorded_backend_audit_v2/inputs.json').read_text())
cfg=OmegaConf.load(old['original_resolved_config'])
tokenizer=AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path,trust_remote_code=True)
with np.load(BASE/'Z/training_signals.npz',allow_pickle=False) as pack:
    ids=pack['input_ids'];full=pack['response_mask'].astype(bool)
response=ids[:,-full.shape[1]:];roles=np.zeros_like(full,dtype=np.uint8)
names={1:'answer_content_overlap',2:'think_content_overlap',3:'EOS',4:'format_role_or_unparsed'}
per_episode=[]
for i,row in enumerate(rows):
    assert time.time()<P['preparation_started_unix']+P['max_seconds_including_preparation']
    ep=episodes[row['seed']];audit=ep['mechanism_audit'];raw=audit['response_ids']
    assert np.array_equal(response[i,:len(raw)],raw)
    assert np.array_equal(full[i,:len(audit['response_mask'])],np.asarray(audit['response_mask'],dtype=bool))
    generated=[s for s in audit['spans'] if s['kind']=='generated'];unparsed=0
    for span in generated:
        begin,end=span['start'],span['end'];assert full[i,begin:end].all()
        tokens=response[i,begin:end].tolist();text=tokenizer.decode(tokens,skip_special_tokens=False)
        match=re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>',text,re.S)
        if not match:unparsed+=1
        lengths=[len(tokenizer.decode(tokens[:k],skip_special_tokens=False)) for k in range(len(tokens)+1)]
        for k,token in enumerate(tokens):
            if token in [151643,151645]:role=3
            elif match and lengths[k+1]>match.start(2) and lengths[k]<match.end(2):role=1
            elif match and lengths[k+1]>match.start(1) and lengths[k]<match.end(1):role=2
            else:role=4
            roles[i,begin+k]=role
    assert np.array_equal(roles[i]>0,full[i]),'generated/observation partition mismatch'
    per_episode.append({'seed':row['seed'],'category':row['category'],'success':bool(ep['traj_success']),
        'generated_tokens':int(full[i].sum()),'answer_overlap_tokens':int((roles[i]==1).sum()),'turns':len(generated),'unparsed_turns':unparsed})
denominator=32*full.sum(-1,keepdims=True)
stats={};arrays={'answer_loss_mask':(roles==1).astype(np.uint8),'full_generation_mask':full.astype(np.uint8),'role_ids':roles}
for branch in ['O_repair1','Z']:
    config=OmegaConf.load(BASE/branch/'resolved.yaml');actor=config.actor_rollout_ref.actor
    assert actor.loss_agg_mode=='token-mean' and actor.ppo_mini_batch_size==32 and actor.ppo_micro_batch_size_per_gpu==1
    assert actor.ppo_epochs==1 and not actor.use_dynamic_bsz and not actor.get('use_rollout_log_probs',False)
    assert actor.entropy_coeff==0 and not actor.use_kl_loss and actor.policy_loss.loss_mode=='vanilla'
    with np.load(BASE/branch/'training_signals.npz',allow_pickle=False) as pack:
        adv=pack['advantages'];assert np.array_equal(pack['input_ids'],ids) and np.array_equal(pack['response_mask'].astype(bool),full)
    coeff=-adv.astype(np.float64)*full/denominator;mass=float(np.abs(coeff).sum());assert mass>0 and np.isfinite(coeff).all()
    role_stats={}
    for number,name in names.items():
        selected=roles==number;values=coeff[selected]
        role_stats[name]={'tokens':int(selected.sum()),'signed_coefficient_sum':float(values.sum()),'absolute_coefficient_mass':float(np.abs(values).sum()),'absolute_mass_fraction':float(np.abs(values).sum()/mass)}
    stats[branch]={'roles':role_stats,'nonanswer_absolute_mass_fraction':1-role_stats['answer_content_overlap']['absolute_mass_fraction'],'absolute_mass_total':mass,'actual_config_checked':True}
    arrays[branch+'_scalar_coefficients']=coeff
    # Small scalar autograd regression through the installed vanilla loss; no model.
    toy=torch.tensor([[-.3,-.4,-.5]],requires_grad=True);advantages=torch.tensor([[1.,-2.,4.]]);mask=torch.tensor([[1.,1.,0.]])
    loss,_=compute_policy_loss_vanilla(toy.detach(),toy,advantages,mask,'token-mean',actor)
    (loss/32).backward();expected=-advantages*mask/(32*mask.sum())
    assert torch.allclose(toy.grad,expected,atol=1e-7,rtol=0)
coverage=all(row['answer_overlap_tokens']>0 for row in per_episode)
gate=coverage and all(s['nonanswer_absolute_mass_fraction']>=.9 for s in stats.values())
result={'status':'COMPLETE_CPU_ONLY','completed_unix':time.time(),'rows':32,'generated_tokens':int(full.sum()),'answer_tokens':int((roles==1).sum()),'every_episode_nonempty_answer_mask':coverage,
    'per_episode':per_episode,'branches':stats,'distinct_loss_region_hypothesis_feasible':bool(gate),
    'coefficient_definition':'-A_it/(32*N_i) on generated tokens. Actual installed one-normalized-mini/one-epoch/use_rollout_log_probsFalse branch sets old_logp=current.detach; r=1 exactly. No entropy/KL/IS contribution in this diagnostic.',
    'CPU_scalar_autograd_fixture_branches':2,'model_forwards':0,'environment_steps':0,'model_backward_passes':0,'optimizer_steps':0,
    'limitations':['Posthoc saved-batch objective support, not independent performance evidence','Coefficient mass is not Jacobian-weighted parameter-gradient mass, Adam influence or Stop causality','Answer-overlap includes punctuation/boundary tokens; no pure-action-logit claim','Old zero-head performance and historical Stop mechanism gates remain failed']}
np.savez_compressed(OUT/'loss_support.npz',**arrays)
(OUT/'result.json').write_text(json.dumps(result,indent=2))
(OUT/'COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),'CPU_only':True,'model_forwards':0,'environment_steps':0,'optimizer_steps':0},indent=2))
assert sum(p.stat().st_size for p in OUT.iterdir())<P['maximum_additional_bytes']
print(json.dumps({k:v for k,v in result.items() if k!='per_episode'},indent=2))
