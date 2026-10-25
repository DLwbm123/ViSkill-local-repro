"""Summarize completed native correctness phases and export only small evidence."""
import os,json,time,math,subprocess,shutil
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);NAME='official_frozenlake_correctness_v1';OUT=ROOT/f'artifacts/reproduction/{NAME}';STATE=ROOT/'state/reproduction'
def read(p):return json.loads(p.read_text())
end=read(STATE/f'{NAME}_EXIT.json');assert end['exit_code']==0 and end['reason'] is None and end['completed_phases']==['train','audit','reload'] and not end['command_line_violations']
assert not (OUT/'FIT_FAILURE.json').exists();train_end=read(OUT/'WORKER_COMPLETED.json');reload_end=read(OUT/'WORKER_RELOAD_COMPLETED.json');audit=read(OUT/'ACTOR_PARAMETER_AUDIT.json');saved=read(OUT/'checkpoint_receipts/step_0001.json');loaded=read(OUT/'RELOAD_APPLIED.json');metrics=read(OUT/'metrics/step_0001.json')['metrics']
assert train_end['last_native_global_batch']==saved['step']==loaded['global_step']==reload_end['restored_global_step']==1
assert train_end['reason']=='planned correctness stop after global batch1' and reload_end['additional_training_updates']==0 and audit['fixed_sample_nonzero_delta']
for key in ['actor/grad_norm','critic/grad_norm','actor/pg_loss','critic/vf_loss']:assert math.isfinite(float(metrics[key]))
rows=[];events=[]
for p in sorted((OUT/'episodes').glob('*.jsonl')):rows.extend(json.loads(s) for s in p.read_text().splitlines())
for p in sorted((OUT/'requests').glob('*.jsonl')):events.extend(json.loads(s) for s in p.read_text().splitlines())
results=[r for r in events if r['event']=='result'];intents=[r for r in events if r['event']=='intent'];errors=[r for r in events if r['event']=='error'];counts=read(OUT/'dispatch_count.json')
assert not errors and len(results)==len(intents)==counts['all'] and len({r['number'] for r in results})==counts['all']
assert sum(r['num_turns'] for r in rows)==counts['rollout'] and sum(r['kind']=='distill' for r in results)==counts['distill']
partitions={}
for mode,step,count in [('val',0,256),('train',1,128),('val',1,256)]:
 group=[r for r in rows if r['mode']==mode and r['step']==step];assert len(group)==count
 if mode=='val':assert {r['seed'] for r in group}==set(range(10001,10257))
 partitions[f'{mode}_{step}']={'episodes':count,'successes':sum(bool(r['traj_success']) for r in group),'retrieved_episodes':sum(bool(r.get('retrieved_skill_id')) for r in group),'model_turns':sum(r['num_turns'] for r in group)}
assert len(rows)==640
library=read(OUT/'library/skill_index.json');snapshot=read(OUT/'checkpoints/global_step_1/online_library/skill_index.json');assert library==snapshot
before={r['seed']:r for r in rows if r['mode']=='val' and r['step']==0};after={r['seed']:r for r in rows if r['mode']=='val' and r['step']==1}
paired={'improved':sum(not before[s]['traj_success'] and bool(after[s]['traj_success']) for s in before),'regressed':sum(bool(before[s]['traj_success']) and not after[s]['traj_success'] for s in before),'scope':'Same official256 panel; both policy weights and skill library changed, so not an isolated parameter-learning effect.'}
# Compare the two already completed pre-update runs once; do not re-evaluate a model.
parent=ROOT/'artifacts/reproduction/official_frozenlake_ppo_v1';parent_rows=[];parent_results=[]
for p in (parent/'episodes').glob('val_0000.jsonl'):parent_rows.extend(json.loads(s) for s in p.read_text().splitlines())
for p in (parent/'requests').glob('val_0000.jsonl'):parent_results.extend(json.loads(s) for s in p.read_text().splitlines() if json.loads(s)['event']=='result')
old={r['seed']:r for r in parent_rows};assert set(old)==set(before)
def first_requests(records):
 out={}
 for r in sorted(records,key=lambda r:r['number']):
  if r['kind']=='rollout':out.setdefault(r['seed'],r)
 return out
old_req=first_requests(parent_results);new_req=first_requests([r for r in results if r['mode']=='val' and r['step']==0])
baseline_repeat={'parent_successes':sum(bool(r['traj_success']) for r in old.values()),'child_successes':partitions['val_0']['successes'],'same_initial_env_states':sum(old[s]['initial_env_state']==before[s]['initial_env_state'] for s in old),'same_first_prompt_token_ID_arrays':sum(old_req[s]['prompt_ids']==new_req[s]['prompt_ids'] for s in old),'same_first_raw_reply':sum(old_req[s]['raw']==new_req[s]['raw'] for s in old),'success_gained':sum(not old[s]['traj_success'] and bool(before[s]['traj_success']) for s in old),'success_lost':sum(bool(old[s]['traj_success']) and not before[s]['traj_success'] for s in old),'scope':'Descriptive comparison of existing recordings; not new model queries or pixel/tensor/weight equality proof. Greedy restart determinism and its cause were not established.'}
receipt=read(STATE/f'{NAME}_receipt.json');residual=[]
for p in psutil.process_iter(['pid','create_time','uids']):
 try:
  if p.info['uids'].real==os.getuid() and p.info['create_time']>=receipt['start_unix']-1 and p.environ().get('TASK_RUN_TOKEN')==receipt['run_token'] and p.status()!=psutil.STATUS_ZOMBIE:residual.append(p.pid)
 except(psutil.NoSuchProcess,psutil.AccessDenied):pass
assert not residual
roots=[ROOT,Path('/data/envs/z0'),Path('/data/tools/z0'),Path('/data/tmp/z0')];usage=sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in roots)
assert usage<receipt['hard_limit_bytes_strict'] and end['end_unix']<end['deadline_unix']
parent_failure=read(parent/'failure_audit.json')
result={'status':'PASS_BOUNDED_NATIVE_TRAIN_SAVE_RELOAD_CORRECTNESS','closed_unix':time.time(),'native_global_training_batches':1,'additional_training_updates_during_reload':0,'metrics':{k:metrics[k] for k in ['actor/pg_loss','actor/grad_norm','critic/vf_loss','critic/grad_norm','actor/lr','critic/lr']},'fixed_actor_delta_sample_elements':audit['sampled_elements'],'fixed_actor_changed_sample_elements':sum(r['nonzero_elements'] for r in audit['rows']),'fixed_actor_sample_matrices':len(audit['rows']),'native_full_checkpoint_allocated_bytes':saved['allocated_checkpoint_bytes'],'native_checkpoint_file_count':len(saved['files']),'native_full_restore_load_returned':loaded,'library_snapshot_values_equal_current':True,'library_entries':len(library.get('skills',[])),'partitions':partitions,'paired_before_after':paired,'prior_preupdate_repeat':baseline_repeat,'child_actual_dispatches':counts,'child_request_errors':len(errors),'child_actual_rollout_output_tokens':sum(len(r['output_ids']) for r in results if r['kind']=='rollout'),'child_actual_distill_output_tokens':sum(len(r['output_ids']) for r in results if r['kind']=='distill'),'parent_failure_cost':parent_failure,'combined_parent_and_child_agent_dispatches':counts['all']+parent_failure['dispatches']['all'],'combined_episode_executions_cost_only':len(rows)+parent_failure['completed_initial_val_episodes']+parent_failure['completed_training_collection_episodes'],'native_training_batch_seconds':metrics['timing_s/step'],'first_native_save_seconds':saved['save_elapsed_seconds'],'child_supervisor_wall_seconds':end['end_unix']-receipt['start_unix'],'child_preparation_to_exit_seconds':end['end_unix']-read(Path(os.environ['PAPER_SOURCE'])/f'configs/reproduction/{NAME}.json')['correctness_preparation_started_unix'],'final_four_root_bytes':usage,'parent_original_baseline_four_root_bytes':receipt['baseline_four_root_bytes'],'peak_additional_bytes_including_parent_failure':end['new_peak_bytes_from_original_baseline'],'peak_four_root_bytes':end['peak_four_root_bytes'],'live_owned_non_zombie_residuals':residual,'exit':end,'limitations':['One global batch only, not301-step paper performance or long-term stability','Actor delta is a fixed36864-element sample, not full parameter equality or direct critic parameter-delta evidence','Native model/optimizer/extra/dataloader load returned; no bytewise equality or continued-training trajectory proof','Weights and online library both changed before restored256 validation; no pure parameter-learning claim','Greedy restart outputs are not all identical; causes and full sampling-training backend distribution equality remain unverified']}
(OUT/'result.json').write_text(json.dumps(result,indent=2))
# Build one atomic allowlisted export. All weights, optimizer files and bulk input IDs remain remote.
dest=OUT/'public_bundle';temp=OUT/'public_bundle_preparing';assert not dest.exists() and not temp.exists();temp.mkdir()
def copy(source,relative):
 target=temp/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for name in ['result.json','cpu_check.json','WORKER_COMPLETED.json','WORKER_RELOAD_COMPLETED.json','ACTOR_PARAMETER_AUDIT.json','RELOAD_APPLIED.json','gpu_before_reload.json','actor_train_actual_loading_info.json','critic_train_actual_loading_info.json','actor_reload_actual_loading_info.json','critic_reload_actual_loading_info.json']:
 copy(OUT/name,name)
for folder in ['metrics','validation_metrics','checkpoint_receipts','episodes','fixed_inputs','library']:
 for p in (OUT/folder).rglob('*'):
  if p.is_file():copy(p,Path(folder)/p.relative_to(OUT/folder))
for p in sorted((OUT/'requests').glob('*.jsonl')):
 small=[]
 for line in p.read_text().splitlines():
  r=json.loads(line)
  if r['event']!='result':continue
  item={k:v for k,v in r.items() if k not in ['prompt_ids','fixed_input_images']};item.update(prompt_token_count=len(r['prompt_ids']),bulk_input_ids_and_images_retained_remote=True);small.append(item)
 target=temp/'requests'/p.name;target.parent.mkdir(exist_ok=True);target.write_text('\n'.join(json.dumps(r) for r in small)+'\n')
receipt.pop('run_token');(temp/'receipt.json').write_text(json.dumps(receipt,indent=2));(temp/'EXIT.json').write_text(json.dumps(end,indent=2))
assert not any(p.suffix in ['.pt','.safetensors','.ckpt','.gz'] for p in temp.rglob('*'));temp.rename(dest)
post_export_usage=sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in roots)
result.update(final_four_root_bytes=post_export_usage,final_additional_bytes_including_small_export=post_export_usage-receipt['baseline_four_root_bytes'],final_budget_measurement_after_small_export=True)
assert post_export_usage<receipt['hard_limit_bytes_strict']
for target in [OUT/'result.json',dest/'result.json']:target.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['parent_failure_cost','native_full_restore_load_returned','limitations']},indent=2));print(json.dumps({'public_files':sum(p.is_file() for p in dest.rglob('*')),'public_bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())}))
