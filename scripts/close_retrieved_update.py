"""Close the stopped native audit or its recorded-input recovery without model calls."""
import os,json,time,math,shutil,subprocess
from pathlib import Path
import psutil
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME=os.environ['TASK_EXPERIMENT_NAME'];OUT=ROOT/'artifacts/reproduction'/NAME;STATE=ROOT/'state/reproduction'
def read(p):return json.loads(p.read_text())
end=read(STATE/f'{NAME}_EXIT.json');receipt=read(STATE/f'{NAME}_receipt.json');P=read(SRC/f'configs/reproduction/{NAME}.json');audits={}
for p in OUT.glob('*_rank*_audit.json'):audits[p.stem]=read(p)
rows=[json.loads(x) for x in (OUT/'episodes/train_0002.jsonl').read_text().splitlines()];assert len(rows)==128 and len({x['seed'] for x in rows})==128
expected=read(OUT/'next_batch_manifest.json')['next_seeds'];assert {x['seed'] for x in rows}==set(expected)
passed=(OUT/'WORKER_COMPLETED.json').exists() and end['exit_code']==0 and end['reason'] is None
summary={}
for role in ['actor','critic']:
 parts=[audits[f'{role}_rank{i}_audit'] for i in [0,1] if f'{role}_rank{i}_audit' in audits]
 if len(parts)==2:
  rr=[x for a in parts for x in a['rows']];assert len(rr)==128 and {x['seed'] for x in rr}==set(expected)
  assert all(a['actual_optimizer_step_calls']==4 and a['changed_sample_elements']>0 for a in parts)
  summary[role]={'complete':True,'rows':128,'retrieved_rows':sum(a['retrieved_rows'] for a in parts),'optimizer_steps_per_rank':[a['actual_optimizer_step_calls'] for a in parts],'sampled_elements':sum(a['sampled_elements'] for a in parts),'changed_sample_elements':sum(a['changed_sample_elements'] for a in parts),'grad_norms':parts[0]['finite_native_metrics'][role+'/grad_norm']}
 else:summary[role]={'complete':False,'available_rank_audits':len(parts)}
if passed:
 assert len(audits)==4 and all(a['complete'] and a['retrieved_rows']>0 for a in summary.values()) and end['completed_phases']==['train']
 assert read(OUT/'metrics/step_0002.json')['step']==2 and read(OUT/'LOAD_APPLIED.json')['global_step']==1
 assert read(OUT/'RECORDED_BATCH_USED.json')['new_model_generation_calls']==0
 assert not list((OUT/'validation_metrics').glob('*')) and not (OUT/'checkpoints').exists()
source=ROOT/'artifacts/reproduction/retrieved_input_ppo_check_v1';events=[json.loads(x) for x in (source/'requests/train_0002.jsonl').read_text().splitlines()];rr=[x for x in events if x['event']=='result'];counts=read(source/'dispatch_count.json')
assert len(rr)==len([x for x in events if x['event']=='intent'])==counts['all'] and not [x for x in events if x['event']=='error']
assert len({x['number'] for x in rr})==counts['all'] and sum(x['num_turns'] for x in rows)==counts['rollout']
owned=[]
for p in psutil.process_iter(['pid','create_time','uids']):
 try:
  if p.info['uids'].real==os.getuid() and p.info['create_time']>=receipt['start_unix']-1 and p.environ().get('TASK_RUN_TOKEN')==receipt['run_token'] and p.status()!=psutil.STATUS_ZOMBIE:owned.append(p.pid)
 except(psutil.NoSuchProcess,psutil.AccessDenied):pass
assert not owned
prior=read(ROOT/'artifacts/reproduction/official_frozenlake_correctness_v1/result.json')
result={'status':'PASS_RECORDED_RETRIEVED_INPUT_NATIVE_PPO' if passed else 'INCOMPLETE_ACTOR_AUDIT_SERIALIZATION_FAILURE','closed_unix':time.time(),'role_summary':summary,'episodes':128,'retrieved_episodes':sum(bool(x['retrieved_skill_id']) for x in rows),'collection_successes_before_update':sum(bool(x['traj_success']) for x in rows),'source_model_generation_dispatches':counts,'new_generation_dispatches_this_recovery':0 if passed else None,'source_request_errors':0,'source_output_tokens':sum(len(x['output_ids']) for x in rr),'combined_parent_and_supplement_model_dispatches_cost_only':prior['combined_parent_and_child_agent_dispatches']+counts['all'],'combined_original_model_collected_episodes_cost_only':prior['combined_episode_executions_cost_only']+128,'CPU_replayed_environment_episodes':128 if passed else 0,'supervisor_wall_seconds':end['end_unix']-receipt['start_unix'],'preparation_to_exit_seconds':end['end_unix']-P['preparation_started_unix'],'private_bank_file_sizes':{p.name:p.stat().st_size for p in OUT.glob('PRIVATE*.pt')},'native_full_reference_checkpoint_retained':(ROOT/'artifacts/reproduction/official_frozenlake_correctness_v1/checkpoints/global_step_1/actor/optim_world_size_2_rank_0.pt').is_file(),'post_step2_model_optimizer_saved':False,'live_owned_non_zombie':owned,'exit':end,'limitations':['Fixed local shard samples do not prove every parameter or exclude every possible bug','No301-step performance or long-run stability claim','Earlier backend numeric negatives and greedy restart differences remain unresolved','Recovery reconstructs saved replies with original environment/processor and checks every prompt/trajectory; not bytewise equality to lost original processed tensors']}
if not passed:result['failure']={'error':'Cannot save multiple tensors or storages that view the same data as different types','location':'new audit-only torch.save(input bank), before native actor update','critic_real_update_completed':True,'actor_native_update_not_completed':True,'updated_critic_live_state_not_saved':True,'partial_private_bank_bytes':792}
else:result['native_metrics']=read(OUT/'metrics/step_0002.json')['metrics']
(OUT/'result.json').write_text(json.dumps(result,indent=2))
dest=OUT/'public_bundle';assert not dest.exists();dest.mkdir()
for p in OUT.iterdir():
 if p.is_file() and p.suffix=='.json' and not p.name.startswith('PRIVATE'):shutil.copy2(p,dest/p.name)
for folder in ['episodes','metrics']:
 for p in (OUT/folder).glob('*'):
  if p.is_file():(dest/folder).mkdir(exist_ok=True);shutil.copy2(p,dest/folder/p.name)
if not passed:
 (dest/'requests').mkdir()
 small=[dict({k:v for k,v in x.items() if k not in ['prompt_ids','fixed_input_images']},prompt_token_count=len(x['prompt_ids']),bulk_inputs_retained_remote=True) for x in rr]
 (dest/'requests/train_0002.jsonl').write_text('\n'.join(json.dumps(x) for x in small)+'\n')
receipt.pop('run_token');(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));(dest/'EXIT.json').write_text(json.dumps(end,indent=2))
roots=[ROOT,Path('/data/envs/z0'),Path('/data/tools/z0'),Path('/data/tmp/z0')];usage=sum(int(subprocess.check_output(['du','-s','-B1','.'],cwd=p,text=True).split()[0]) for p in roots)
assert usage<receipt['hard_limit_bytes_strict'] and end['end_unix']<P['deadline_unix']
result.update(final_four_root_bytes=usage,combined_supplement_new_bytes=usage-P['substage_baseline_four_root_bytes'],original_parent_additional_bytes=usage-P['storage_baseline_four_root_bytes'])
for p in [OUT/'result.json',dest/'result.json']:p.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['native_metrics','exit']},indent=2))
