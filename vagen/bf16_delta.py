"""Exact BF16 inference parameter XOR; encoded arrays are private weights."""
import time
import numpy as np
import torch

def retain_state_delta(state,model,base,folder,limit,deadline):
    """Stream one BF16 inference delta; never write a full model checkpoint."""
    import json,re,zipfile
    from pathlib import Path
    from contextlib import ExitStack
    from safetensors import safe_open
    base=Path(base);folder=Path(folder);folder.mkdir(exist_ok=False)
    assert model.config.tie_word_embeddings
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    state.pop('lm_head.weight')
    index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
    mapping=type(model)._checkpoint_conversion_mapping;canonical={}
    for key,file in index.items():
        name=key
        for pattern,replacement in mapping.items():name=re.sub(pattern,replacement,name)
        assert name not in canonical;canonical[name]=(key,file)
    assert set(state)==set(canonical)-{'lm_head.weight'}
    entries=[];arrays=0;changed=0
    archive=folder/'PRIVATE_LEARNED_BF16_DELTA.npz'
    with ExitStack() as stack:
        readers={f:stack.enter_context(safe_open(str(base/f),framework='pt',device='cpu')) for f in set(index.values())}
        writer=LimitedWriter(archive,limit,deadline)
        try:
            with zipfile.ZipFile(writer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as pack:
                for i,key in enumerate(sorted(state)):
                    assert time.time()<deadline
                    source,file=canonical[key];original=readers[file].get_tensor(source)
                    target=state[key].detach().cpu().to(torch.bfloat16)
                    assert original.dtype==torch.bfloat16 and original.shape==target.shape
                    delta=encode(original,target);count=int(np.count_nonzero(delta));name=f'k{i:04d}' if count else None
                    if count:
                        with pack.open(name+'.npy','w',force_zip64=True) as stream:np.lib.format.write_array(stream,delta,allow_pickle=False)
                        arrays+=1
                    changed+=count
                    entries.append({'target_key':key,'source_key':source,'source_file':file,'shape':list(delta.shape),'dtype':'bfloat16','array':name})
        finally:writer.close()
    schema={'format':'bf16_xor_npz_v1','complete':True,'base_model':'models/b0','tied_alias_omitted':'lm_head.weight','entries':entries}
    (folder/'PRIVATE_PARAMETER_SCHEMA.json').write_text(json.dumps(schema))
    return {'covered_tensors':len(entries),'stored_delta_arrays':arrays,'changed_BF16_elements':changed,'private_archive_bytes':archive.stat().st_size,'inference_only':True,'operational_reload_verified':False}

def bits(tensor):
    assert tensor.dtype==torch.bfloat16 and tensor.device.type=='cpu'
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16)

def encode(base,target):
    assert base.shape==target.shape
    return np.bitwise_xor(bits(base),bits(target))

def decode(base,delta):
    assert delta.dtype==np.uint16 and tuple(delta.shape)==tuple(base.shape)
    raw=np.bitwise_xor(bits(base),delta)
    return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)

class LimitedWriter:
    def __init__(self,path,limit,deadline):
        self.file=open(path,'w+b');self.limit=limit;self.deadline=deadline
    def write(self,value):
        if time.time()>=self.deadline:raise TimeoutError('original preparation-inclusive encoding deadline')
        if self.file.tell()+len(value)>self.limit:raise RuntimeError('private encoded-weight write stop')
        return self.file.write(value)
    def tell(self):return self.file.tell()
    def seek(self,*args):return self.file.seek(*args)
    def flush(self):return self.file.flush()
    def close(self):return self.file.close()

def apply_to_model(model,archive,schema):
    """Restore inference BF16 values over the retained b0 model, not optimizer state."""
    assert schema['format']=='bf16_xor_npz_v1' and schema['complete']
    assert model.config.tie_word_embeddings
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    state=model.state_dict();expected={entry['target_key'] for entry in schema['entries']}
    assert set(state)-{'lm_head.weight'}==expected
    with np.load(archive,allow_pickle=False) as pack,torch.no_grad():
        for entry in schema['entries']:
            tensor=state[entry['target_key']];assert list(tensor.shape)==entry['shape']
            if entry['array'] is not None:
                tensor.copy_(decode(tensor.detach().cpu().to(torch.bfloat16),pack[entry['array']]).to(tensor.dtype))
    return {'applied_parameter_tensors':sum(e['array'] is not None for e in schema['entries']),
            'inference_only':True,'full_FP32_optimizer_resume_verified':False}
