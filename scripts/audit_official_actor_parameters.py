"""Read nine predeclared small actor parameter slices; no model/GPU operation."""
import os,json,math,time
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(os.environ['TASK_ROOT']);NAME=os.environ['TASK_EXPERIMENT_NAME'];OUT=ROOT/f'artifacts/reproduction/{NAME}';base=ROOT/'models/b0';after=OUT/'checkpoints/global_step_1/actor/huggingface'
assert not (OUT/'ACTOR_PARAMETER_AUDIT.json').exists()
metrics=json.loads((OUT/'metrics/step_0001.json').read_text())['metrics']
def number(v):
    if isinstance(v,list):assert len(v)==1;return number(v[0])
    return float(v)
for k in ['actor/grad_norm','critic/grad_norm']:
    assert k in metrics and math.isfinite(number(metrics[k])) and number(metrics[k])>0
cfg=json.loads((base/'config.json').read_text());n=cfg.get('text_config',cfg)['num_hidden_layers'];layers=[0,n//2,n-1];types=['self_attn.q_proj.weight','self_attn.o_proj.weight','mlp.down_proj.weight']
def mapping(folder):
    index=json.loads((folder/'model.safetensors.index.json').read_text());result={}
    for key,file in index['weight_map'].items():
        canonical=key.replace('model.language_model.','model.')
        assert canonical not in result;result[canonical]=(key,file)
    return result
left,right=mapping(base),mapping(after);rows=[]
for i in layers:
    for kind in types:
        canonical=f'model.layers.{i}.{kind}';lk,lf=left[canonical];rk,rf=right[canonical]
        with safe_open(base/lf,framework='pt',device='cpu') as f:
            view=f.get_slice(lk);shape=view.get_shape();before=view[:2,:2048].float()
        with safe_open(after/rf,framework='pt',device='cpu') as f:
            view=f.get_slice(rk);assert view.get_shape()==shape;new=view[:2,:2048].float()
        assert before.numel()==new.numel()==4096 and torch.isfinite(new).all()
        delta=new-before;rows.append({'parameter':canonical,'slice':'first2rows_first2048columns','sampled_elements':delta.numel(),'finite':True,'nonzero_elements':int(torch.count_nonzero(delta)),'max_abs_delta':float(delta.abs().max()),'L2_delta':float(delta.norm())})
result={'status':'FIXED_ACTOR_SLICES_AND_NATIVE_GRADIENT_METRICS_PASS','completed_unix':time.time(),'rows':rows,'sampled_elements':sum(x['sampled_elements'] for x in rows),'fixed_sample_nonzero_delta':any(x['nonzero_elements']>0 for x in rows),'actor_grad_norm':number(metrics['actor/grad_norm']),'critic_grad_norm':number(metrics['critic/grad_norm']),'skill_metrics':{k:v for k,v in metrics.items() if k.startswith('skill/')},'new_model_forward_backward_calls':0,'new_GPU_calls':0,'new_optimizer_updates':0,'scope':'Small deterministic sample confirms actor changed vs promoted b0; not full parameter equality or direct critic parameter-delta verification. Native actor/critic save/load is checked separately.'}
assert result['fixed_sample_nonzero_delta'];(OUT/'ACTOR_PARAMETER_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
