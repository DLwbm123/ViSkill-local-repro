"""One CPU-only, fixed-Z bounded parameter encoding experiment."""
import os,json,time,signal,re,zipfile,tempfile
from pathlib import Path
from contextlib import ExitStack
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from safetensors import safe_open
from vagen.bf16_delta import bits,encode,decode,LimitedWriter,apply_to_model

ROOT=Path(os.environ['TASK_ROOT']);NAME='bf16_delta_feasibility_v1'
P=json.loads((ROOT/f'src/configs/smoke/{NAME}.json').read_text());OUT=ROOT/f'artifacts/component_tests/{NAME}'
deadline=min(P['absolute_total_deadline'],P['preparation_started_unix']+P['max_seconds_including_preparation'])
assert time.time()+30<deadline
def timeout(*args):raise TimeoutError('original preparation-inclusive deadline')
signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(deadline-time.time())))
archive=OUT/'PRIVATE_LEARNED_BF16_DELTA.npz';assert not archive.exists()
entries=[];changed=0;total=0;arrays=0;status='FAILED';error=None;fixture_pass=False;cap_pass=False;apply_pass=False
try:
    a=torch.from_numpy(np.array([0,32768,16256,49024,128,1,32639,65407],dtype=np.uint16).view(np.int16)).view(torch.bfloat16)
    b=torch.from_numpy(np.array([0,32768,16257,49023,0,32769,32512,65280],dtype=np.uint16).view(np.int16)).view(torch.bfloat16)
    assert np.array_equal(bits(decode(a,encode(a,b))),bits(b));fixture_pass=True
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        writer=LimitedWriter(Path(tmp)/'fixture',128,deadline);writer.write(b'x'*64)
        try:writer.write(b'x'*65)
        except RuntimeError:cap_pass=True
        else:raise AssertionError('writer cap did not reject oversized write')
        writer.close()
        from types import SimpleNamespace
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed=torch.nn.Embedding(2,4);self.head=torch.nn.Linear(4,2,bias=False)
                self.embed.weight=torch.nn.Parameter(a.reshape(2,4).float().clone());self.head.weight=self.embed.weight
                self.config=SimpleNamespace(tie_word_embeddings=True)
            def get_input_embeddings(self):return self.embed
            def get_output_embeddings(self):return self.head
            def state_dict(self,*args,**kwargs):return {'embed.weight':self.embed.weight.detach(),'lm_head.weight':self.head.weight.detach()}
        toy=Toy();path=Path(tmp)/'toy.npz';np.savez_compressed(path,k0000=encode(a.reshape(2,4),b.reshape(2,4)))
        apply_to_model(toy,path,{'format':'bf16_xor_npz_v1','complete':True,'entries':[{'target_key':'embed.weight','shape':[2,4],'array':'k0000'}]})
        assert np.array_equal(bits(toy.embed.weight.detach().to(torch.bfloat16)),bits(b.reshape(2,4)))
        assert toy.embed.weight is toy.head.weight;apply_pass=True
    assert json.loads((ROOT/'artifacts/component_tests/actor_loss_support_audit_v1/result.json').read_text())['distinct_loss_region_hypothesis_feasible']
    base=ROOT/P['base'];index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
    conversion=P['official_checkpoint_conversion_mapping']
    canonical={}
    for key,file in index.items():
        name=key
        for pattern,replacement in conversion.items():name=re.sub(pattern,replacement,name)
        assert name not in canonical;canonical[name]=(key,file)
    with ExitStack() as stack:
        target=stack.enter_context(safe_open(str(ROOT/P['target']),framework='pt',device='cpu'))
        readers={name:stack.enter_context(safe_open(str(base/name),framework='pt',device='cpu')) for name in set(index.values())}
        keys=sorted(target.keys());assert set(keys)==set(canonical)-{'lm_head.weight'}
        for key in keys:
            source,file=canonical[key];a=readers[file].get_slice(source);b=target.get_slice(key)
            assert a.get_shape()==b.get_shape() and a.get_dtype()==b.get_dtype()=='BF16'
        writer=LimitedWriter(archive,P['write_stop_bytes'],deadline)
        try:
            with zipfile.ZipFile(writer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
                for i,key in enumerate(keys):
                    assert time.time()<deadline
                    source,file=canonical[key];original=readers[file].get_tensor(source);current=target.get_tensor(key)
                    delta=encode(original,current);count=int(np.count_nonzero(delta));name=f'k{i:04d}' if count else None
                    if count:
                        with z.open(name+'.npy','w',force_zip64=True) as stream:np.lib.format.write_array(stream,delta,allow_pickle=False)
                        arrays+=1
                    changed+=count;total+=delta.size
                    entries.append({'target_key':key,'source_key':source,'source_file':file,'shape':list(delta.shape),'dtype':'bfloat16','array':name})
                    if i%50==0:(OUT/'progress.json').write_text(json.dumps({'completed_tensors':i+1,'total_tensors':len(keys),'private_archive_bytes':writer.tell(),'changed_elements':changed,'elapsed_seconds':time.time()-P['preparation_started_unix']}))
                    del original,current,delta
        finally:writer.close()
    schema={'format':'bf16_xor_npz_v1','complete':True,'base_model':P['base'],'tied_alias_omitted':'lm_head.weight','entries':entries}
    (OUT/'PRIVATE_PARAMETER_SCHEMA.json').write_text(json.dumps(schema))
    with zipfile.ZipFile(archive) as z:assert len(z.namelist())==arrays
    status='COMPLETE_FORMAT_FEASIBLE'
except BaseException as exc:error=repr(exc)
finally:
    signal.alarm(0)
    result={'status':status,'error':error,'completed_unix':time.time(),'preparation_started_unix':P['preparation_started_unix'],'deadline_unix':deadline,
        'covered_tensors':len(entries),'stored_delta_arrays':arrays,'total_elements_processed':total,'changed_BF16_elements':changed,
        'private_archive_bytes':archive.stat().st_size if archive.exists() else 0,'small_CPU_bit_roundtrip_passed':fixture_pass,'bounded_writer_fixture_passed':cap_pass,'tiny_tied_module_apply_fixture_passed':apply_pass,
        'model_forwards':0,'environment_steps':0,'optimizer_steps':0,'full_model_roundtrip_or_reload_verified':False,
        'scope':'Parameter encoding only. Archive and parameter schema are PRIVATE weights, never public numerical evidence. No model byte-identity/hash check and no training-resume claim.'}
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if status!='COMPLETE_FORMAT_FEASIBLE':raise RuntimeError(error)
