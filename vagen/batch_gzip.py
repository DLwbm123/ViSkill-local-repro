"""Fixed lossless CPU input container; contents remain private runtime data."""
import copy,gzip
import numpy as np
import torch
from verl import DataProto
from vagen.bf16_delta import LimitedWriter

def clone_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:clone_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [clone_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(clone_cpu(v) for v in value)
    if isinstance(value,np.ndarray):
        if value.dtype!=object:return value.copy()
        result=np.empty(value.shape,dtype=object)
        for index in np.ndindex(value.shape):result[index]=clone_cpu(value[index])
        return result
    return copy.deepcopy(value)

def save_batch(batch,path,limit,deadline):
    from pathlib import Path
    assert not Path(path).exists(),'never replace a completed or partial container'
    payload={'format':'dataproto_gzip_torch_v1','batch':{k:clone_cpu(v) for k,v in batch.batch.items()},
             'non_tensor_batch':clone_cpu(batch.non_tensor_batch),'meta_info':clone_cpu(batch.meta_info)}
    writer=LimitedWriter(path,limit,deadline)
    try:
        with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=writer,mtime=0) as stream:torch.save(payload,stream)
    finally:writer.close()

def load_batch(path):
    with gzip.open(path,'rb') as stream:payload=torch.load(stream,map_location='cpu',weights_only=False)
    assert payload['format']=='dataproto_gzip_torch_v1'
    return DataProto.from_dict(tensors=payload['batch'],non_tensors=payload['non_tensor_batch'],meta_info=payload['meta_info'])
