"""Bounded teacher-forced scoring and saved-batch audit. No rollout or updates."""
import json, os, time
from pathlib import Path
import numpy as np
import torch
import ray
from PIL import Image
from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from vagen.ray_trainer import RayPPOTrainer
from vagen.main_ppo_skill import TaskRunner
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
from verl.workers.rollout.sglang_rollout import async_sglang_server as native

ROOT=Path(os.environ['TASK_ROOT'])
OUT=ROOT/'artifacts/component_tests/recorded_backend_audit_v2'

def save(name, value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))

class CheckService(native.SGLangHttpServer.__ray_metadata__.modified_class):
    async def launch_server(self,*args,**kwargs):
        import sglang.srt.entrypoints.engine as engine
        from vagen.sglang_prefill_only import run_scheduler
        engine.run_scheduler_process=run_scheduler
        self.config.engine_kwargs['sglang']['disable_overlap_schedule']=True
        self.config.engine_kwargs['sglang']['disable_radix_cache']=True
        return await super().launch_server(*args,**kwargs)

    async def score_saved(self,index,request,image_paths):
        from sglang.srt.managers.io_struct import GenerateReqInput
        used=getattr(self,'_used',set());assert index not in used and len(used)<4
        used.add(index);self._used=used
        assert not (OUT/f'native/{index}_started.json').exists(),'refuse replaying a dispatched score'
        save(f'native/{index}_started.json',{'index':index,'new_tokens_requested':0})
        images=[Image.open(p).convert('RGB') for p in image_paths]
        ids=request['prompt_ids']+request['output_ids']
        query=GenerateReqInput(rid=f'check_{index}',input_ids=ids,image_data=images,
            sampling_params={'max_new_tokens':0,'temperature':1.0,'top_p':1.0,'repetition_penalty':1.0},
            return_logprob=True,logprob_start_len=len(request['prompt_ids'])-1,top_logprobs_num=5)
        output=await self.tokenizer_manager.generate_request(query,None).__anext__()
        save(f'native/{index}.json',output)
        assert output['meta_info']['completion_tokens']==0,'native generated tokens despite zero-generation request'
        assert not output.get('output_ids',[]),'native output ids are nonempty'
        return output

class CheckWorker(AsyncActorRolloutRefWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from transformers import set_seed
        set_seed(int(os.environ['CHECK_RUN_SEED']))

class ValueWorker(CriticWorker):
    def _build_critic_model_optimizer(self,config):
        import verl.utils.model as models
        original=models.load_valuehead_model
        stored=json.loads(Path(os.environ['CHECK_HEAD']).read_text())['head_values']
        def load(*args,**kwargs):
            model=original(*args,**kwargs);found={}
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in stored and not p.is_meta:
                        v=stored[name];p.copy_(torch.tensor(v['values'],dtype=p.dtype).reshape(v['shape']))
                        found[name]={'shape':list(p.shape),'values':p.detach().float().cpu().flatten().tolist()}
            if self.rank==0:
                assert found==stored,'saved initial value head was not restored exactly'
                save('initial_head_check.json',{'full_small_head_matches_original':True,'parameters':sum(len(x['values']) for x in found.values()),'backbone':'same retained b0; no full-model identity claim'})
            return model
        models.load_valuehead_model=load
        try:return super()._build_critic_model_optimizer(config)
        finally:models.load_valuehead_model=original

class CheckTrainer(RayPPOTrainer):
    def make_batch(self,rows):
        tensors=[];mm=[]
        for row in rows:
            prompt=row['prompt_ids'];response=row['response_ids'];mask=row['response_mask']
            assert len(prompt)<=8192 and len(response)<=4096 and len(response)==len(mask)
            self.tokenizer.padding_side='left'
            p=self.tokenizer.pad({'input_ids':prompt},padding='max_length',max_length=8192,return_tensors='pt',return_attention_mask=True)
            self.tokenizer.padding_side='right'
            r=self.tokenizer.pad({'input_ids':response},padding='max_length',max_length=4096,return_tensors='pt',return_attention_mask=True)
            for obj in [p,r]:
                for k in ['input_ids','attention_mask']:
                    if obj[k].dim()==1:obj[k]=obj[k].unsqueeze(0)
            ids=torch.cat([p['input_ids'],r['input_ids']],1)
            attention=torch.cat([p['attention_mask'],r['attention_mask']],1)
            images=[Image.open(f).convert('RGB') for f in row['image_paths']]
            text=self.tokenizer.decode(ids.squeeze(0),skip_special_tokens=True)
            inputs=dict(self.processor(text=[text],images=images,return_tensors='pt'))
            inputs.pop('input_ids',None);inputs.pop('attention_mask',None)
            assert int(inputs['image_grid_thw'].shape[0])==len(images)
            vision=get_rope_index(self.processor,input_ids=ids.squeeze(0),image_grid_thw=inputs.get('image_grid_thw'),
                video_grid_thw=inputs.get('video_grid_thw'),second_per_grid_ts=inputs.get('second_per_grid_ts'),attention_mask=attention.squeeze(0)).unsqueeze(0)
            textpos=torch.ones((1,ids.shape[1]),dtype=torch.long)
            valid=attention[0].bool();textpos[0,valid]=torch.arange(valid.sum())
            position=torch.cat([textpos.unsqueeze(0),vision],1)
            response_mask=torch.tensor([mask+[0]*(4096-len(mask))])*r['attention_mask']
            tensors.append({'prompts':p['input_ids'],'responses':r['input_ids'],'input_ids':ids,
                'attention_mask':attention,'position_ids':position,'response_mask':response_mask})
            mm.append(inputs)
        tensor={k:torch.cat([x[k] for x in tensors]) for k in tensors[0]}
        objects=np.empty(len(mm),dtype=object);objects[:]=mm
        return DataProto.from_dict(tensors=tensor,non_tensors={'multi_modal_inputs':objects},meta_info={'temperature':1.0})

    def fit(self):
        manifest=json.loads((OUT/'inputs.json').read_text());episodes=manifest['episodes']
        ids=manifest['score_episode_ids'];full=[episodes[k] for k in ids];prefix=[]
        for row in full:
            cut=row['turns'][1]['end']
            prefix.append(dict(row,response_ids=row['response_ids'][:cut],response_mask=row['response_mask'][:cut],image_paths=row['turns'][1]['image_paths']))
        score_batch=self.make_batch(full+prefix)
        retained=[json.loads(p.read_text()) for p in (OUT/'native').glob('*.json') if not p.name.endswith('_started.json')]
        save('cost.json',{'native_requests_started':len(list((OUT/'native').glob('*_started.json'))),'score_actor_samples':0,'audit_actor_samples':0,'audit_critic_samples':0,'generated_tokens':sum(x['meta_info']['completion_tokens'] for x in retained),'environment_steps':0,'optimizer_steps':0,'admissions':0})
        manager=self.async_rollout_manager;manager.wake_up()
        save('hybrid_wake.json',{'returned':True,'time_unix':time.time(),'replicas':len(manager.server_handles),'weights':'initial retained b0 via production hybrid sync; no full-weight identity claim'})
        native_outputs=[]
        for i,case in enumerate(manifest['cases']):
            row=episodes[case['episode_id']];turn=row['turns'][case['turn']]
            req=json.loads(Path(case['request']).read_text())
            previous=OUT/f'native/{i}.json'
            if previous.exists():
                # Preserve and reuse the one completed pre-repair input score;
                # its unwanted newline remains a recorded contract failure.
                assert (OUT/f'native/{i}_started.json').exists()
                native_outputs.append(json.loads(previous.read_text()))
            else:
                native_outputs.append(ray.get(manager.server_handles[i%len(manager.server_handles)].score_saved.remote(i,req,turn['image_paths'])))
        manager.sleep()
        save('score_actor_started.json',{'samples':8,'time_unix':time.time()})
        actual=self.actor_rollout_wg.compute_log_prob(score_batch).batch['old_log_probs'].cpu()
        torch.save({'log_probs':actual,'response_mask':score_batch.batch['response_mask']},OUT/'score_actor.pt')
        comparisons=[]
        for i,(case,native_output) in enumerate(zip(manifest['cases'],native_outputs,strict=True)):
            row=episodes[case['episode_id']];turn=row['turns'][case['turn']]
            req=json.loads(Path(case['request']).read_text());n=len(req['output_ids'])
            raw=native_output['meta_info']['input_token_logprobs'];tail=raw[-n:]
            assert [int(x[1]) for x in tail]==req['output_ids'],'teacher-forced response token ids not aligned'
            values=torch.tensor([x[0] for x in tail],dtype=torch.float32)
            assert len(values)==n and torch.isfinite(values).all()
            episode_index=ids.index(case['episode_id']);a=actual[episode_index,turn['start']:turn['end']].float()
            assert torch.isfinite(a).all()
            diff=a-values;absolute=diff.abs()
            result={'case':i,'seed':row['seed'],'group':row['group'],'turn':case['turn']+1,
                'tokens':n,'native_completion_tokens':native_output['meta_info']['completion_tokens'],
                'zero_generation_contract_pass':native_output['meta_info']['completion_tokens']==0,
                'native_prompt_tokens':native_output['meta_info']['prompt_tokens'],
                'expected_prefill_tokens':len(req['prompt_ids'])+n,
                'max_abs_nats':float(absolute.max()),'mean_abs_nats':float(absolute.mean()),'p95_abs_nats':float(torch.quantile(absolute,.95)),
                'actor_logprob':a.tolist(),'native_logprob':values.tolist(),'token_ids':req['output_ids'],
                'token_pieces':[self.tokenizer.decode([t],skip_special_tokens=False) for t in req['output_ids']],
                'ratio_actor_over_native':diff.exp().tolist()}
            if case['turn']==1:
                p=actual[episode_index+4,turn['start']:turn['end']].float();d=(p-values).abs()
                result['prefix_actor_max_abs_nats']=float(d.max());result['prefix_actor_mean_abs_nats']=float(d.mean())
                result['full_vs_prefix_actor_max_abs_nats']=float((p-a).abs().max())
            result['operational_gate_pass']=result['max_abs_nats']<=.05 and result['mean_abs_nats']<=.005
            comparisons.append(result)
        generated=sum(x['native_completion_tokens'] for x in comparisons)
        save('score_results.json',{'cases':comparisons,'all_eight_probability_gate_pass':all(x['operational_gate_pass'] for x in comparisons),
            'strict_zero_generation_contract_pass':generated==0,'native_returned_generated_tokens':generated,
            'retained_incident':'First completed native max_new_tokens0 request returned one unused newline; original hard failure retained, not rescored. Remaining requests use explicit skip_sample score-only adapter with synchronous uncached prefill.',
            'intermediate_images':'deterministic state reconstruction with saved endpoint matches, not preserved historical intermediate tensors',
            'scope':'b0 four episodes, first/second turn only; theta4 unavailable'})
        save('cost.json',{'native_requests_started':9,'native_prefill_forward_samples':8,'native_dispatch_failed_before_forward':1,'original_eight_dispatch_contract_pass':False,'score_actor_samples':8,'audit_actor_samples':0,'audit_critic_samples':0,'generated_tokens':generated,'environment_steps':0,'optimizer_steps':0,'admissions':0})
        training=[episodes[k] for k in manifest['training_episode_ids']]
        batch=self.make_batch(training)
        save('audit_actor_started.json',{'samples':32,'time_unix':time.time()})
        old=self.actor_rollout_wg.compute_log_prob(batch).batch['old_log_probs'].cpu()
        torch.save(old,OUT/'audit_actor.pt')
        save('audit_critic_started.json',{'samples':32,'time_unix':time.time()})
        values=self.critic_wg.compute_values(batch).batch['values'].cpu()
        torch.save(values,OUT/'audit_critic.pt')
        return self.finish_signal_audit(training,batch,old,values,generated)

    def finish_signal_audit(self,training,batch,old,values,generated):
        from verl.trainer.ppo.core_algos import compute_gae_advantage_return
        from verl.utils.torch_functional import masked_whiten
        original_values_dtype=str(values.dtype)
        values=values.float()  # lossless BF16 -> FP32, matching production FP32 reward promotion
        mask=batch.batch['response_mask'];reward=torch.zeros_like(values,dtype=torch.float32);timed=torch.zeros_like(values,dtype=torch.float32)
        for i,row in enumerate(training):
            generated_spans=list(row['turns'])
            assert len(generated_spans)==len(row['env_rewards'])
            reward[i,generated_spans[-1]['end']-1]=row['reward']
            for turn,r in zip(generated_spans,row['env_rewards'],strict=True):timed[i,turn['end']-1]=r
        adv,returns=compute_gae_advantage_return(reward,values,mask,1.,1.)
        zero_adv,zero_returns=compute_gae_advantage_return(reward,torch.zeros_like(values),mask,1.,1.)
        timed_adv,timed_returns=compute_gae_advantage_return(timed,values,mask,1.,1.)
        constant=float((values*mask).sum()/mask.sum())
        shifted=masked_whiten(returns-(values-constant),mask)
        active=mask.bool();shift_error=float((shifted-adv).abs()[active].max())
        return_error=float((returns-reward.sum(-1,keepdim=True)).abs()[active].max())
        assert return_error<=.001 and shift_error<=.0001
        payload={'values':values,'returns':returns,'raw_advantages':returns-values,'advantages':adv,
            'old_log_probs':old,'response_mask':mask,'terminal_rewards':reward,'per_turn_rewards':timed,
            'constant_shift_advantages':shifted,'zero_baseline_advantages':zero_adv,
            'per_turn_advantages':timed_adv,'per_turn_returns':timed_returns,
            'input_ids':batch.batch['input_ids'],'position_ids':batch.batch['position_ids'],'attention_mask':batch.batch['attention_mask']}
        assert all(torch.isfinite(t).all() for t in payload.values())
        torch.save(payload,OUT/'first_batch_signals.pt')
        turns=[]
        for i,row in enumerate(training):
            for t,turn in enumerate(row['turns']):
                begin,end=turn['start'],turn['end'];tokens=row['response_ids'][begin:end]
                decoded=[self.tokenizer.decode(tokens[:k],skip_special_tokens=True) for k in range(len(tokens)+1)]
                text=decoded[-1];start=text.find('<answer>');finish=text.find('</answer>')
                action=text[start+8:finish].strip() if start>=0 and finish>start else ''
                positions=[begin+k for k in range(len(tokens)) if start>=0 and finish>start and len(decoded[k+1])>start+8 and len(decoded[k])<finish]
                turns.append({'seed':row['seed'],'category':row['category'],'success':row['success'],'turn':t+1,'answer':action,
                    'immediate_reward':row['env_rewards'][t],'generated_start':begin,'generated_end':end,
                    'answer_token_positions':positions,'answer_token_ids':[row['response_ids'][p] for p in positions],
                    'answer_old_log_probs':[float(old[i,p]) for p in positions],
                    'answer_values':[float(values[i,p]) for p in positions],
                    'answer_raw_advantages':[float(returns[i,p]-values[i,p]) for p in positions],
                    'answer_advantages':[float(adv[i,p]) for p in positions],
                    'answer_zero_baseline_advantages':[float(zero_adv[i,p]) for p in positions],
                    'answer_per_turn_advantages':[float(timed_adv[i,p]) for p in positions],
                    'initial_pg_loss_coefficients_original_mini8':[-float(adv[i,p])/int(mask[i].sum())/8 for p in positions]})
        save('training_signal_audit.json',{'rows':32,'successes':sum(r['success'] for r in training),'original_forward_values_dtype':original_values_dtype,'analysis_reward_dtype':'torch.float32',
            'return_max_error':return_error,'constant_value_shift':constant,'constant_shift_whitening_max_error':shift_error,
            'generated_token_count':int(mask.sum()),'value_mean_generated':float(values[active].mean()),
            'return_mean_generated':float(returns[active].mean()),'turns':turns,
            'limitations':['Reconstructed input, intermediate historical tensors absent','Initial b0 and saved full small critic head only; no theta4 recovery','No backward or parameter update: advantage sign and initial coefficients do not prove actual probability changes','Original mini8 has sequential updates; this audit does not replay those updates','Whitening weights generated tokens, while micro1 actor loss averages each episode before accumulation']})
        save('cost.json',{'native_requests_started':9,'native_prefill_forward_samples':8,'native_dispatch_failed_before_forward':1,'original_eight_dispatch_contract_pass':False,'score_actor_samples':8,'audit_actor_samples':32,'audit_critic_samples':32,'generated_tokens':generated,'environment_steps':0,'optimizer_steps':0,'admissions':0})
        save('COMPLETE.json',{'completed_unix':time.time(),'score_cases':8,'training_rows':32,'new_training':False,'original_zero_generation_contract_pass':generated==0,'retained_incidental_generated_tokens':generated,'original_eight_dispatch_contract_pass':False,'native_dispatch_attempts':9,'native_forward_samples':8})

class CheckTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self,config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(CheckWorker)
        return CheckWorker,RayWorkerGroup
    def add_critic_worker(self,config):
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.Critic]=ray.remote(ValueWorker)
    def run(self,config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer=CheckTrainer
        native.SGLangHttpServer=ray.remote(num_cpus=1)(CheckService)
        return super().run(config)
