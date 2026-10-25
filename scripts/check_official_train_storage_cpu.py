"""Bounded metadata/meta-device check of released dense-PPO recovery storage."""
import os,json,time,runpy,math,struct
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='official_train_checkpoint_cpu_v1'
P=json.loads((SRC/f'configs/reproduction/{NAME}.json').read_text());OUT=ROOT/f'artifacts/reproduction/{NAME}';OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
assert time.time()<P['deadline_unix']
import torch,ray
from omegaconf import OmegaConf
from transformers import AutoConfig,AutoModelForVision2Seq,AutoModelForTokenClassification,AutoModelForCausalLM
from verl.utils.config import validate_config
from verl.workers.config.optimizer import build_optimizer
from verl.utils.model import load_valuehead_model
_,original=runpy.run_path(str(SRC/'scripts/release_eval_config.py'))['config']()
def restore(value):
    if isinstance(value,dict):return {k:restore(v) for k,v in value.items()}
    if isinstance(value,list):return [restore(v) for v in value]
    if isinstance(value,str):return value.replace(str(ROOT/'models/r0'),str(ROOT/'models/b0'))
    return value
cfg=OmegaConf.create(restore(original));validate_config(cfg,use_reference_policy=False,use_critic=True)
assert cfg.algorithm.adv_estimator=='gae' and not cfg.trainer.val_only
assert cfg.data.train_batch_size==128 and cfg.actor_rollout_ref.actor.ppo_mini_batch_size==32
assert cfg.actor_rollout_ref.actor.optim.lr==1e-6 and cfg.critic.optim.lr==1e-5
assert cfg.actor_rollout_ref.model.path==cfg.critic.model.path==str(ROOT/'models/b0')
assert cfg.skill_system.reward.mode=='skill_guided' and cfg.skill_system.reward.guided_weight==.8
assert cfg.skill_system.distill.mode=='self' and not cfg.skill_system.cold_start.enable
assert not cfg.actor_rollout_ref.actor.freeze_vision_tower and not cfg.critic.model.get('freeze_vision_tower',False)
assert cfg.actor_rollout_ref.model.get('lora_rank',0)==cfg.critic.model.get('lora_rank',0)==0
assert cfg.actor_rollout_ref.actor.optim.optimizer==cfg.critic.optim.optimizer=='AdamW'
assert cfg.actor_rollout_ref.actor.optim.optimizer_impl==cfg.critic.optim.optimizer_impl=='torch.optim'
base=ROOT/'models/b0';hfcfg=AutoConfig.from_pretrained(base,local_files_only=True);assert hfcfg.tie_word_embeddings
hfcfg.torch_dtype=torch.float32;hfcfg._attn_implementation='eager'
with torch.device('meta'):
    actor=AutoModelForVision2Seq.from_config(hfcfg)
actor.tie_weights();assert actor.get_input_embeddings().weight is actor.get_output_embeddings().weight
calls=[];saved={}
def meta_pretrained(cls,*args,**kw):
    calls.append(cls.__name__)
    assert Path(kw['pretrained_model_name_or_path'])==base
    config=kw['config'];config.torch_dtype=kw['torch_dtype'];config._attn_implementation='eager'
    model=cls.from_config(config)
    model.tie_weights();return model
classes=[AutoModelForVision2Seq,AutoModelForTokenClassification,AutoModelForCausalLM]
for cls in classes:saved[cls]=cls.__dict__.get('from_pretrained');cls.from_pretrained=classmethod(meta_pretrained)
try:
    criticcfg=AutoConfig.from_pretrained(base,local_files_only=True);criticcfg.num_labels=1;criticcfg.classifier_dropout=0.;criticcfg.hidden_dropout='0';criticcfg.summary_dropout_prob=0.
    with torch.device('meta'):critic=load_valuehead_model(str(base),torch.float32,criticcfg,False)
finally:
    for cls,descriptor in saved.items():
        if descriptor is None:delattr(cls,'from_pretrained')
        else:setattr(cls,'from_pretrained',descriptor)
assert all(p.is_meta for model in [actor,critic] for p in model.parameters())
# Both production workers explicitly cast the complete module to FP32 before FSDP.
actor.to(dtype=torch.float32);critic.to(dtype=torch.float32)
assert all(p.dtype==torch.float32 for model in [actor,critic] for p in model.parameters())
def count(model):
    return {'independent_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'decoder_layer_parameters':sum(p.numel() for n,p in model.named_parameters() if '.layers.' in n),'value_head_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith('v_head.'))}
a=count(actor);c=count(critic)
index=json.loads((base/'model.safetensors.index.json').read_text());elements=0;dtype_counts={}
for name in sorted(set(index['weight_map'].values())):
    with (base/name).open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
    for name,item in header.items():
        if name=='__metadata__':continue
        size=math.prod(item['shape']);elements+=size;dtype_counts[item['dtype']]=dtype_counts.get(item['dtype'],0)+size
assert elements==a['independent_parameters']==3754622976
# A one-scalar CPU fixture inspects the actual configured AdamW moment dtypes only.
x=torch.nn.Parameter(torch.ones(1,dtype=torch.float32));optim=build_optimizer([x],cfg.actor_rollout_ref.actor.optim);x.grad=torch.ones_like(x);optim.step();moments={k:{'dtype':str(v.dtype),'bytes_per_element':v.element_size()} for k,v in optim.state[x].items() if k in ['exp_avg','exp_avg_sq']};assert set(moments)=={'exp_avg','exp_avg_sq'} and all(v['bytes_per_element']==4 for v in moments.values())
model_bytes=4*(a['independent_parameters']+c['independent_parameters']);dense_moments=8*(a['trainable_parameters']+c['trainable_parameters']);decoder_moments=8*(a['decoder_layer_parameters']+c['decoder_layer_parameters']);hf_actor_bytes=4*a['independent_parameters']
raw_core=model_bytes+dense_moments;raw_conservative=model_bytes+decoder_moments
settings={'algorithm':cfg.algorithm.adv_estimator,'train_batch':cfg.data.train_batch_size,'minibatch':cfg.actor_rollout_ref.actor.ppo_mini_batch_size,'actor_lr':cfg.actor_rollout_ref.actor.optim.lr,'critic_lr':cfg.critic.optim.lr,'actor_optimizer':cfg.actor_rollout_ref.actor.optim.optimizer,'critic_optimizer':cfg.critic.optim.optimizer,'actor_model_dtype_default':'fp32','critic_model_dtype':cfg.critic.model.fsdp_config.get('model_dtype','fp32'),'actor_checkpoint_save_contents':list(cfg.actor_rollout_ref.actor.checkpoint.save_contents),'critic_checkpoint_save_contents':list(cfg.critic.checkpoint.save_contents),'checkpoint_frequency':cfg.trainer.save_freq,'max_actor_checkpoints':cfg.trainer.max_actor_ckpt_to_keep,'max_critic_checkpoints':cfg.trainer.max_critic_ckpt_to_keep,'total_training_steps':cfg.trainer.total_training_steps,'validation_frequency':cfg.trainer.test_freq,'val_before_train':cfg.trainer.val_before_train,'agent_workers':cfg.actor_rollout_ref.rollout.agent.num_workers,'data_max_prompt':cfg.data.max_prompt_length,'data_max_response':cfg.data.max_response_length,'full_generated_loss_no_answer_mask':True,'cold_start':False,'online_initial_library':'new empty library, not released265 or old diagnostic library','proposed_execution_GPU_count':cfg.trainer.n_gpus_per_node,'official_script_GPU_count':4}
assert settings['actor_checkpoint_save_contents']==['model','hf_model','optimizer','extra'] and settings['critic_checkpoint_save_contents']==['model','optimizer','extra']
assert raw_conservative>P['single_experiment_storage_limit_bytes_strict']
result={'status':'ORIGINAL_FULL_TRAINING_CHECKPOINT_STORAGE_BLOCKED','completed_unix':time.time(),'preparation_started_unix':P['prepared_from_heartbeat_unix'],'deadline_unix':P['deadline_unix'],'settings':settings,'actor_meta':a,'critic_meta':c,'critic_meta_class':type(critic).__name__,'critic_loader_attempt_classes':calls,'base_serialized_elements':elements,'base_serialized_dtype_elements':dtype_counts,'scalar_CPU_AdamW_moment_fixture':moments,'scalar_CPU_fixture_steps':1,'new_actual_model_forward_backward_calls':0,'new_GPU_calls':0,'new_native_env_steps':0,'new_real_actor_or_critic_optimizer_steps':0,'model_state_FP32_tensor_payload_bytes':model_bytes,'full_dense_FP32_Adam_moment_payload_bytes':dense_moments,'conservative_model_plus_only_decoder_moments_payload_bytes':raw_conservative,'standard_model_plus_dense_moments_payload_bytes':raw_core,'additional_actor_HF_export_tensor_payload_bytes':hf_actor_bytes,'standard_with_actor_HF_export_tensor_payload_bytes':raw_core+hf_actor_bytes,'current_experiment_limit_bytes_strict':P['single_experiment_storage_limit_bytes_strict'],'project_headroom_after_last_closure_bytes':P['current_project_write_stop_bytes']-P['last_closed_four_root_bytes'],'full_training_GPU_memory_and_throughput':'UNMEASURED','full_checkpoint_serialization_tested':False,'budget_interpretation':'Analytic uncompressed tensor payload from actual meta architectures and configured FP32 AdamW, excluding serialization overhead/RNG/dataloader/skill library/logs; not a measured file size. Decoder-only moment count deliberately excludes vision/embeddings/value head as a conservative dense-connected-backbone bound. FSDP sharding changes per-rank split, not per-server total. Compression has no assumed saving.','decision':'Do not launch official full training with this raw complete checkpoint under existing50GB and85GB limits; prepare reviewable budget/retention options without silent algorithm or recovery changes.'}
assert not torch.cuda.is_initialized() and not ray.is_initialized() and time.time()<P['deadline_unix']
(OUT/'result.json').write_text(json.dumps(result,indent=2));assert sum(p.stat().st_size for p in OUT.iterdir())<P['maximum_additional_output_bytes']
print(json.dumps(result,indent=2))
