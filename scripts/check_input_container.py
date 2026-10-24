"""One existing saved C32 container, fixed gzip1, no model or environment."""
import os,json,time,signal,tempfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path(os.environ['TASK_ROOT']);name='input_container_feasibility_v1'
p=json.loads((root/f'src/configs/smoke/{name}.json').read_text());out=root/f'artifacts/component_tests/{name}'
deadline=min(p['preparation_started_unix']+p['max_seconds_including_preparation'],1792891370.446,1792899507)
def stop(*args):raise TimeoutError('original input-container preparation-inclusive deadline')
signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(deadline-time.time())))
assert not out.exists() and time.time()+60<deadline;out.mkdir()
import torch,numpy as np,psutil
from verl import DataProto
from vagen.batch_gzip import save_batch,load_batch

def metadata(batch):
    mm=batch.non_tensor_batch.get('multi_modal_inputs',[])
    return {'rows':len(batch),'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in batch.batch.items()},
            'non_tensor_keys':sorted(batch.non_tensor_batch),'MM_rows':len(mm),'MM_tensors':[{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in row.items() if isinstance(v,(torch.Tensor,np.ndarray))} for row in mm]}

fixture_pass=False;actual_pass=False;status='FAILED';error=None;source_bytes=None;original_metadata=None;known_samples=None
path=out/'PRIVATE_INPUT.pt.gz'
try:
    raw=torch.zeros(256,dtype=torch.uint8);ids=raw[:64].view(torch.int64).reshape(2,4);rewards=raw[64:96].view(torch.float32).reshape(2,4)
    ids.copy_(torch.arange(8).reshape(2,4));rewards.copy_(torch.arange(8).reshape(2,4)/3)
    mm=np.empty(2,dtype=object)
    for i in range(2):mm[i]={'pixel_values':torch.tensor([[i+.25,i+.5]],dtype=torch.float32),'image_grid_thw':torch.tensor([[1,2,2]])}
    toy=DataProto.from_dict(tensors={'input_ids':ids,'rewards':rewards},non_tensors={'multi_modal_inputs':mm,'seed':np.array([1,2])},meta_info={'temperature':1.})
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        fixture=Path(temporary)/'fixture.gz';save_batch(toy,fixture,1000000,deadline);restored=load_batch(fixture)
        assert metadata(toy)==metadata(restored) and restored.meta_info==toy.meta_info
        assert all(torch.equal(toy.batch[k],restored.batch[k]) for k in toy.batch.keys())
        assert all(torch.equal(mm[i][k],restored.non_tensor_batch['multi_modal_inputs'][i][k]) for i in range(2) for k in mm[i])
    fixture_pass=True
    source=root/'artifacts/component_tests/value_head_single_step_v1/C/shared_batch.pt';source_bytes=source.stat().st_size
    original=torch.load(source,map_location='cpu',weights_only=False);assert len(original)==32
    original_metadata=metadata(original)
    counts=original.batch['response_mask'].sum(-1).tolist()
    known_samples=original.batch['input_ids'][[0,-1]][:,[0,8191,8192,-1]].tolist()
    save_batch(original,path,p['write_stop_bytes'],deadline)
    del original
    actual=load_batch(path)
    assert metadata(actual)==original_metadata and actual.batch['response_mask'].sum(-1).tolist()==counts
    assert actual.batch['input_ids'][[0,-1]][:,[0,8191,8192,-1]].tolist()==known_samples
    actual_pass=True;status='COMPLETE_CPU_INPUT_USABILITY'
except BaseException as exc:error=repr(exc)
finally:
    signal.alarm(0)
    result={'status':status,'error':error,'completed_unix':time.time(),'preparation_started_unix':p['preparation_started_unix'],'deadline_unix':deadline,
        'small_mixed_dtype_byte_view_MM_fixture_passed':fixture_pass,'single_existing_C32_basic_usability_passed':actual_pass,
        'source_container_bytes':source_bytes,'private_gzip_bytes':path.stat().st_size if path.exists() else 0,'metadata':original_metadata,
        'actual_full_byte_identity_verified':False,'source_unchanged_read_only':True,'model_forwards':0,'model_backwards':0,'model_optimizer_steps':0,'environment_steps':0,
        'main_argv':psutil.Process().cmdline(),'scope':'One fixed gzip1 encoding; existing C32 data only. Actual reload checks metadata, mask counts and eight fixed IDs, not a full value or byte comparison. Input container includes private runtime context and stays remote; only stats are public.'}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='metadata'}))
if not actual_pass:raise RuntimeError(error)
