"""Exact BF16 inference parameter XOR; encoded arrays are private weights."""
import time
import numpy as np
import torch

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
