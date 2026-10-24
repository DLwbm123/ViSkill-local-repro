"""Saved-state coverage and tiny optimizer algebra; never a model experiment."""
import os,json,time,signal,copy,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from omegaconf import OmegaConf
from vagen.skills.solver import FrozenLakeSolver,FROZENLAKE_DIRECTIONS
from verl.workers.config.optimizer import FSDPOptimizerConfig,build_optimizer
from verl.utils.config import omega_conf_to_dataclass

root=Path(os.environ['TASK_ROOT']);name='retention_cpu_screen_v1'
p=json.loads((root/f'src/configs/smoke/{name}.json').read_text());out=root/f'artifacts/component_tests/{name}'
assert not out.exists();out.mkdir()
deadline=min(p['preparation_started_unix']+p['max_seconds_including_preparation'],p['candidate_preparation_started_unix']+14400,1792899507)
def stop(*args):raise TimeoutError('original CPU preparation-inclusive deadline')
signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(deadline-time.time())))
assert time.time()<deadline
solver=FrozenLakeSolver();directions=solver.ACTION_ID_TO_NAME
def features(state):
    s=copy.deepcopy(state);s['player_pos']=tuple(s['player_pos'])
    if s.get('is_slippery') is not False or s.get('slip_prob')!=0:
        return {'bfs':None,'unknown_reason':'deterministic dynamics not established'}
    distance=solver.distance(s);row,col=s['player_pos'];inside=[];safe=[];optimal=[]
    for act,(dr,dc) in FROZENLAKE_DIRECTIONS.items():
        nr,nc=row+dr,col+dc
        if 0<=nr<len(s['desc']) and 0<=nc<len(s['desc'][0]):
            inside.append(directions[act])
            if s['desc'][nr][nc]!='H':
                safe.append(directions[act]);other=dict(s,player_pos=(nr,nc));d=solver.distance(other)
                if d is not None and distance is not None and d+1==distance:optimal.append(directions[act])
    return {'bfs':distance,'in_bounds_initial_actions':sorted(inside),'safe_initial_actions':sorted(safe),'optimal_initial_actions':sorted(optimal),'optimal_signature':'+'.join(sorted(optimal)) or 'unknown_or_terminal','unknown_reason':None if distance is not None else 'unreachable'}
fixture={'desc':[['S','F'],['F','G']],'player_pos':[0,0],'is_slippery':False,'slip_prob':0.}
assert features(fixture)['optimal_initial_actions']==['down','right'] and features(fixture)['bfs']==2
assert features(dict(fixture,desc=[['S','H'],['H','G']]))['bfs'] is None

old=root/'artifacts/component_tests/value_head_single_step_v1';new=root/'artifacts/component_tests/answer_region_step_v1'
train=json.loads((old/'data_manifest.json').read_text())['splits']['train'];dev=json.loads((new/'data_manifest.json').read_text())['splits']['dev']
episodes={json.loads(f.read_text())['seed']:json.loads(f.read_text()) for f in (old/'C/episodes').glob('*/episode.json')}
records=json.loads((new/'result.json').read_text())['records'];policies={k:{r['seed']:r for r in records if r['policy']==k} for k in ['b0','Z','A']}
train_rows=[];dev_rows=[]
for split,rows in [('train',train),('dev',dev)]:
    for r in rows:
        f=features(r['state']);assert f['bfs']==r['bfs']
        item={'seed':r['seed'],'category':r['category'],**f}
        if split=='train':
            ep=episodes[r['seed']];item.update(collection_success=bool(ep['traj_success']),retrieved_skill_id=ep['retrieved_skill_id'],executed_actions=ep['trajectory_actions']);train_rows.append(item)
        else:
            item['policies']={k:{field:policies[k][r['seed']][field] for field in ['success','quality','actions','stop_turns','has_stop','hole','terminal_reason','retrieved_skill_id']} for k in policies}
            item['A_vs_b0_changed_fields']=[field for field in item['policies']['A'] if item['policies']['A'][field]!=item['policies']['b0'][field]]
            dev_rows.append(item)
assert len(train_rows)==len(dev_rows)==32 and not ({r['seed'] for r in train_rows}&{r['seed'] for r in dev_rows})
def summary(rows):
    return {'episodes':len(rows),'BFS_counts':dict(collections.Counter(str(r['bfs']) for r in rows)),'optimal_signature_episode_counts':dict(collections.Counter(r['optimal_signature'] for r in rows)),'optimal_action_episode_memberships':{a:sum(a in r['optimal_initial_actions'] for r in rows) for a in ['up','down','left','right']},'unknown_feature_rows':sum(r['unknown_reason'] is not None for r in rows)}
successful=[r for r in train_rows if r['collection_success']]
signatures={r['optimal_signature'] for r in train_rows};successful_signatures={r['optimal_signature'] for r in successful}
for r in dev_rows:
    r['optimal_signature_seen_in_C32']=r['optimal_signature'] in signatures
    r['optimal_signature_seen_in_successful_C32']=r['optimal_signature'] in successful_signatures
coverage={'train_all':summary(train_rows),'train_successful':summary(successful),'dev_all':summary(dev_rows),'dev_conditional':summary([r for r in dev_rows if r['category']=='conditional']),'dev_natural':summary([r for r in dev_rows if r['category']=='natural_multi']),
 'training_executed_action_events':dict(collections.Counter(a for r in train_rows for a in r['executed_actions'])),'successful_training_executed_action_events':dict(collections.Counter(a for r in successful for a in r['executed_actions'])),
 'all_dev_rows':dev_rows,'all_train_rows':train_rows,'causal_or_deployable_routing_claim':False}

cfg=OmegaConf.load(old/'Z/resolved.yaml');optim=omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.optim,dataclass_type=FSDPOptimizerConfig)
assert optim.optimizer=='AdamW' and optim.optimizer_impl=='torch.optim' and optim.lr==1e-6
optimizer_fixtures=[]
for dtype in [torch.float32,torch.float64]:
    for label,g in [('above_epsilon',[.1,-.2,.3]),('epsilon_sensitive',[1e-11,-2e-10,3e-9])]:
        before=torch.tensor([.5,-.3,.2],dtype=dtype);gradient=torch.tensor(g,dtype=dtype);deltas=[];norms=[]
        for scale in [torch.ones(3,dtype=dtype),torch.tensor([.1,10.,.01],dtype=dtype)]:
            param=torch.nn.Parameter(before.clone());opt=build_optimizer([param],optim);assert len(opt.state)==0
            param.grad=gradient*scale;norm=torch.nn.utils.clip_grad_norm_([param],float(cfg.actor_rollout_ref.actor.grad_clip));opt.step()
            deltas.append((param.detach()-before).tolist());norms.append(float(norm))
        maximum=max(abs(a-b) for a,b in zip(*deltas))
        optimizer_fixtures.append({'dtype':str(dtype),'case':label,'initial_gradients':g,'positive_diagonal_scaling':[.1,10.,.01],'pre_clip_norms':norms,'base_update':deltas[0],'scaled_update':deltas[1],'maximum_update_difference':maximum,'difference_over_learning_rate':maximum/optim.lr})
theta=torch.tensor([.2,-.3,.4],dtype=torch.float64,requires_grad=True);reference=theta.detach().clone()
ewc=(torch.tensor([1.,2.,3.])*torch.square(theta-reference)).sum()/2;ewc.backward();ewc_grad=float(theta.grad.abs().max())
theta2=reference.clone().requires_grad_();target=torch.softmax(reference,dim=0);kl=(target*(torch.log_softmax(reference,dim=0)-torch.log_softmax(theta2,dim=0))).sum();kl.backward();kl_grad=float(theta2.grad.abs().max())
assert ewc_grad==0 and kl_grad<1e-12
result={'status':'COMPLETE_CPU_ONLY','completed_unix':time.time(),'coverage':coverage,'optimizer_config':OmegaConf.to_container(cfg.actor_rollout_ref.actor.optim,resolve=True),'optimizer_fixtures':optimizer_fixtures,'initial_same_reference_EWC_max_gradient':ewc_grad,'initial_same_reference_KL_max_gradient':kl_grad,'model_forwards':0,'model_backwards':0,'model_optimizer_steps':0,'environment_steps':0,'tiny_CPU_optimizer_steps':8,'tiny_CPU_autograd_backwards':2,
 'limitations':['Posthoc64 saved states and96 policy outcomes; no independent new evaluation','BFS optimal actions are privileged diagnostics, not a deployment routing policy','First-action signature coverage does not establish board/state/action-history coverage or causality','First AdamW fixture includes an epsilon-sensitive counterexample; actual model gradient-coordinate distribution is not known','Identical-reference KL/EWC at initialization has zero first gradient in the fixture; no whole-model regularizer was tested','Old answer-region investment gate remains FAIL; no new GPU protocol, mask/LR/seed search or launch follows automatically']}
payload=json.dumps(result,indent=2).encode();assert len(payload)<p['maximum_additional_bytes'] and time.time()<deadline
(out/'result.json').write_bytes(payload);signal.alarm(0)
print(json.dumps({'status':result['status'],'bytes':len(payload),'coverage':{k:v for k,v in coverage.items() if k not in ['all_dev_rows','all_train_rows']},'optimizer_fixtures':optimizer_fixtures,'EWC_gradient':ewc_grad,'KL_gradient':kl_grad}))
