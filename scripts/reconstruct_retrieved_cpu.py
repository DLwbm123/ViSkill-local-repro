"""Reconstruct the full existing batch on CPU and reject any request/trajectory mismatch."""
import os,runpy,asyncio,json,types,time,tempfile,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,ray
from transformers import AutoTokenizer,AutoProcessor
from vagen.gym_agent_dataset import AgenticDataset
from vagen import retrieved_replay as replay
from vagen.retrieved_update_check import snapshot
from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase,_agent_loop_registry
ROOT,SRC,OUT,P=replay.ROOT,replay.SRC,replay.OUT,replay.P
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'cpu_check.json').exists()
cfg=runpy.run_path(str(SRC/'scripts/run_retrieved_update_check.py'))['config']()
assert not cfg.actor_rollout_ref.rollout.calculate_log_probs
with tempfile.TemporaryDirectory(dir=OUT) as td:
 p=Path(td)/'fixture.pt';storage=torch.arange(16,dtype=torch.uint8);mixed={'int':storage.view(torch.int32),'float':storage.view(torch.float32)}
 try:torch.save(mixed,p)
 except RuntimeError as exc:assert 'different types' in str(exc)
 else:raise AssertionError('Must reproduce dtype-alias save failure')
 torch.save(snapshot(mixed),p);restored=torch.load(p,weights_only=False)
 for k,v in mixed.items():assert restored[k].dtype==v.dtype and torch.equal(restored[k],v)
source=replay.SOURCE
shutil.copytree(ROOT/'artifacts/reproduction/official_frozenlake_correctness_v1/checkpoints/global_step_1/online_library',OUT/'library',dirs_exist_ok=True)
for folder in ['episodes','requests','fixed_inputs','metrics','validation_metrics','checkpoint_receipts']:(OUT/folder).mkdir(exist_ok=True)
shutil.copy2(source/'episodes/train_0002.jsonl',OUT/'episodes/train_0002.jsonl');shutil.copy2(source/'next_batch_manifest.json',OUT/'next_batch_manifest.json')
expected=json.loads((OUT/'next_batch_manifest.json').read_text())['next_seeds'];dataset=AgenticDataset(cfg.data.train_files,cfg.data)
items={int(dataset[i]['seed']):dataset[i] for i in range(len(dataset))};tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/b0',trust_remote_code=True);processor=AutoProcessor.from_pretrained(ROOT/'models/b0',trust_remote_code=True)
_agent_loop_registry['gym_agent']={'_target_':'vagen.retrieved_replay.ReplayLoop'};replay.ReplayLoop.checks=[]
worker=types.SimpleNamespace(config=cfg,tokenizer=tokenizer,processor=processor,server_manager=None,reward_router_address=None)
worker._run_agent_loop=types.MethodType(AgentLoopWorkerBase._run_agent_loop,worker)
params=dict(temperature=cfg.actor_rollout_ref.rollout.temperature,top_p=cfg.actor_rollout_ref.rollout.top_p,repetition_penalty=1.,logprobs=False)
async def main():
 outputs=[]
 for i,seed in enumerate(expected):
  outputs.append(await worker._run_agent_loop(params,dict(step=2,sample_index=i,rollout_n=0,validate=False),**items[seed]))
 return outputs
outputs=asyncio.run(main());data=AgentLoopWorkerBase._postprocess(worker,outputs);data.meta_info.pop('metrics');data.meta_info['timing']={'replayed_existing_requests':0.}
bank={'tensors':{k:snapshot(v) for k,v in data.batch.items()},'non_tensors':snapshot(data.non_tensor_batch),'meta_info':snapshot(data.meta_info)}
torch.save(bank,OUT/'PRIVATE_REPLAY_GENERATION.pt')
assert len(replay.ReplayLoop.checks)==128 and sum(x['matched_prompt_and_sampling_requests'] for x in replay.ReplayLoop.checks)==415
assert not ray.is_initialized() and not torch.cuda.is_initialized()
result={'status':'CPU_READY','completed_unix':time.time(),'mixed_dtype_storage_failure_reproduced_and_snapshot_roundtrip_pass':True,'reconstructed_episodes':128,'all415_recorded_prompt_IDs_sampling_and_image_counts_match':True,'response_IDs_masks_retrieval_initial_states_actions_positions_distill_rewards_match':True,'new_model_generation_calls':0,'real_model_forward_backward_calls':0,'GPU_calls':0,'CPU_environment_replay_episodes':128,'private_bank_bytes':(OUT/'PRIVATE_REPLAY_GENERATION.pt').stat().st_size,'per_episode_checks':replay.ReplayLoop.checks,'limitation':'Reconstruction uses original environment and processor plus recorded replies, not bytewise proof of the lost original processed pixel tensors.'}
replay.write(OUT/'cpu_check.json',result);print(json.dumps({k:v for k,v in result.items() if k!='per_episode_checks'}))
