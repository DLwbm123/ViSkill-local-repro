"""Summarize a completed official256 pass and export small synthetic evidence once."""
import os,json,math,time,shutil
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);SRC=Path(os.environ['PAPER_SOURCE']);NAME='released_frozenlake_eval_v1';OUT=ROOT/f'artifacts/reproduction/{NAME}'
end=json.loads((ROOT/f'state/reproduction/{NAME}_EXIT.json').read_text());assert end['exit_code']==0 and end['reason'] is None and not end['command_line_violations']
complete=json.loads((OUT/'COMPLETE.json').read_text());rows=[json.loads(p.read_text()) for p in (OUT/'episodes').glob('*.json')];assert len(rows)==256 and {r['seed'] for r in rows}==set(range(10001,10257))
requests=[json.loads(p.read_text()) for p in (OUT/'requests').glob('*.json') if '.intent.' not in p.name];assert len(requests)==int((OUT/'dispatch_count.txt').read_text())
rollout=[r for r in requests if r['kind']=='rollout'];distill=[r for r in requests if r['kind']=='distill'];assert len(rollout)==sum(r['num_turns'] for r in rows)<=1280 and len(distill)<=256
assert all(r['sampling'].get('temperature')==0 for r in rollout)
loading=json.loads((OUT/'actual_loading_info.json').read_text());assert not any(loading[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
assert complete['optimizer_steps']==complete['admissions']==0 and complete['library_json_values_unchanged']
k=sum(bool(r['traj_success']) for r in rows);n=256;rate=k/n;z=1.959963984540054;den=1+z*z/n;center=(rate+z*z/(2*n))/den;width=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
result=dict(complete,status='COMPLETE_OFFICIAL_RELEASE_BENCHMARK_CHECK',closed_unix=time.time(),successes=k,denominator=n,success_rate=rate,gap_to_paper_reference=rate-.84,nominal_binomial_Wilson95_interval=[center-width,center+width],interval_scope='Descriptive nominal binomial interval; not training-seed uncertainty, statistical equivalence or a from-scratch reproduction claim.',all_256_records=sorted(rows,key=lambda r:r['seed']),exit=end,actual_loading_info=loading,limitations=['Published checkpoint model card does not identify cold-start variant or training seed','2 RTX PRO 5000 Blackwell GPUs and Triton/SDPA, not paper H100 hardware','Released code reward differs from paper state-progress description','Initialization/warmup and any internal backend retries are separate from counted agent dispatches'])
prior=OUT.with_name(NAME+'_initialization_failure')/'initialization_failure_audit.json'
if prior.exists():result['initialization_failure']=json.loads(prior.read_text())
(OUT/'result.json').write_text(json.dumps(result,indent=2))
dest=OUT/'public_bundle';dest.mkdir(exist_ok=False)
def copy(path,relative):
 target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
if prior.exists():copy(prior,'initialization_failure_audit.json')
for name in ['result.json','COMPLETE.json','metrics.json','actual_loading_info.json','cpu_check.json','evaluation_started.json']:
 copy(OUT/name,name)
for p in (OUT/'episodes').glob('*.json'):copy(p,Path('episodes')/p.name)
# All generated replies remain public; bulk input images/token prefixes stay remote.
for r in requests:
 small={key:r[key] for key in ['number','seed','kind','request_id','sampling','output_ids','raw','dispatched_unix','returned_unix']};small.update(prompt_token_count=len(r['prompt_ids']),image_count=len(r['images']),raw_input_images_and_prompt_ids_retained_remote=True)
 target=dest/'requests'/f"{r['number']:04d}.json";target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(small))
# Two predetermined seed examples, independent of success, illustrate actual inputs.
for seed in [10001,10002]:
 first=min((r for r in rollout if r['seed']==seed),key=lambda r:r['number'])
 for rel in first['images']:copy(OUT/rel,Path('fixed_input_examples')/Path(rel).name)
receipt=json.loads((ROOT/f'state/reproduction/{NAME}_receipt.json').read_text());receipt.pop('run_token',None);(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));(dest/'EXIT.json').write_text(json.dumps(end,indent=2))
assert not any(p.suffix in ['.pt','.gz','.safetensors','.ckpt'] or p.name.startswith('PRIVATE_') for p in dest.rglob('*'))
files=[p for p in dest.rglob('*') if p.is_file()];print(json.dumps({'successes':k,'n':n,'success_rate':rate,'gap_to_reference':rate-.84,'Wilson95':[center-width,center+width],'rollout_dispatches':len(rollout),'post_episode_distill_dispatches':len(distill),'public_files':len(files),'public_bytes':sum(p.stat().st_size for p in files),'from_scratch_training_reproduced':False}))
