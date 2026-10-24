"""Score-only adapter for SGLang0.5.2's max_new_tokens=0 sampling leak.

Uses its existing skip_sample forward and input-logprob processor. No installed
package is edited; ordinary generation is delegated unchanged.
"""
import time

def install(scheduler_class):
    if getattr(scheduler_class,'_zero_score_installed',False):return
    original_run=scheduler_class.run_batch
    original_process=scheduler_class.process_batch_result_prefill

    def run(self,batch):
        if not batch.is_prefill_only:return original_run(self,batch)
        assert not self.enable_overlap,'zero-score path requires synchronous prefill'
        forward=self.tp_worker.forward_batch_generation
        def without_sampling(model_batch,*args,**kwargs):
            # This flag belongs to ScheduleBatch, not ModelWorkerBatch.
            assert batch.is_prefill_only
            kwargs['skip_sample']=True
            return forward(model_batch,*args,**kwargs)
        self.tp_worker.forward_batch_generation=without_sampling
        try:return original_run(self,batch)
        finally:self.tp_worker.forward_batch_generation=forward

    def process(self,batch,result,launch_done=None):
        if not batch.is_prefill_only:return original_process(self,batch,result,launch_done)
        assert not self.enable_overlap and self.is_generation
        assert result.next_token_ids is None,'score-only forward sampled a token'
        output=result.logits_output
        output.input_token_logprobs=tuple(output.input_token_logprobs.tolist())
        offset=0
        for i,req in enumerate(batch.reqs):
            assert req.sampling_params.max_new_tokens==0 and not req.output_ids
            assert req.is_chunked<=0,'unexpected chunked input; refuse unsupported score path'
            count=result.extend_input_len_per_req[i]-result.extend_logprob_start_len_per_req[i]
            self.add_input_logprob_return_values(i,req,output,offset,count,last_prefill_chunk=True)
            offset+=count
            req.check_finished()
            assert req.finished() and not req.output_ids
            self.tree_cache.cache_finished_req(req)
            req.time_stats.completion_time=time.time()
        self.stream_output(batch.reqs,batch.return_logprob,None)

    scheduler_class.run_batch=run
    scheduler_class.process_batch_result_prefill=process
    scheduler_class._zero_score_installed=True

def run_scheduler(*args,**kwargs):
    from sglang.srt.managers.scheduler import Scheduler,run_scheduler_process
    install(Scheduler)
    return run_scheduler_process(*args,**kwargs)

def self_check():
    """Verify zero-sampling delegation without loading a model or an environment."""
    from types import SimpleNamespace as S
    import torch
    sampled=[];finished=[];streams=[]
    class Fake:
        enable_overlap=False;is_generation=True
        def __init__(self):
            def forward(batch,skip_sample=False):
                sampled.append(not skip_sample)
                return S(next_token_ids=None if skip_sample else torch.tensor([9]),
                    logits_output=S(input_token_logprobs=torch.tensor([-.2,-.3])),
                    extend_input_len_per_req=[2],extend_logprob_start_len_per_req=[0])
            self.tp_worker=S(forward_batch_generation=forward)
            self.tree_cache=S(cache_finished_req=lambda req:finished.append(req))
        def run_batch(self,batch):return self.tp_worker.forward_batch_generation(S())
        def process_batch_result_prefill(self,*args):raise AssertionError('wrong processor')
        def add_input_logprob_return_values(self,i,req,output,offset,count,last_prefill_chunk):
            assert count==2 and last_prefill_chunk
            req.logprobs=output.input_token_logprobs
        def stream_output(self,reqs,return_logprob,skip):streams.append([len(r.output_ids) for r in reqs])
    install(Fake);worker=Fake()
    req=S(sampling_params=S(max_new_tokens=0),output_ids=[],is_chunked=0,time_stats=S())
    req.check_finished=lambda:setattr(req,'done',True);req.finished=lambda:getattr(req,'done',False)
    batch=S(is_prefill_only=True,reqs=[req],return_logprob=True)
    result=worker.run_batch(batch);worker.process_batch_result_prefill(batch,result)
    assert sampled==[False] and streams==[[0]] and len(finished)==1
    worker.run_batch(S(is_prefill_only=False))
    assert sampled==[False,True]
    return {'score_samples_generated':0,'zero_output_metadata_ready':True,'ordinary_forward_delegated':True}
