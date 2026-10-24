"""Finite predeclared diagnostic; retains the original total clock."""
import json,os,pathlib,signal,subprocess,time
root=pathlib.Path(os.environ['TASK_ROOT']);state=root/'state/components';prefix=os.environ['TASK_ATTEMPT']
assert prefix == 'directional_native_loop_v1'
receipt=state/f'{prefix}_receipt.json'
if receipt.exists():raise RuntimeError('attempt exists; do not duplicate')
original=json.loads((root/'state/setup_receipt.json').read_text())
protocol=json.loads((root/'src/configs/smoke/directional_native_loop_v1.json').read_text())
assert original['started_unix']==protocol['original_total_clock_start']
start=time.time();deadline=min(original['started_unix']+86400,start+900)
write_limit=0  # resolved below from actual baseline or original correction receipt
def usage():
 return sum(int(x.split()[0]) for x in subprocess.check_output(['du','-s','-B1',*original['paths']],text=True).splitlines())
write_limit=min(protocol['project_write_stop_bytes'],usage()+protocol['maximum_additional_bytes'])
if time.time()>=deadline or usage()>=write_limit:raise RuntimeError('budget exhausted')
free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<28000:raise RuntimeError('insufficient GPU4 memory')
env=dict(os.environ,ENTRY=str(root/'src/scripts/check_directional_native_loop.py'),CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',RAY_USAGE_STATS_ENABLED='0',HF_HOME=str(root/'cache/hf'),TMPDIR='/data/local/tmp/z0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',XDG_CACHE_HOME='/data/local/tools/z0/cache/xdg',TORCH_EXTENSIONS_DIR='/data/local/tools/z0/cache/torch_extensions',TRITON_CACHE_DIR='/data/local/tools/z0/cache/triton',FLASHINFER_WORKSPACE_BASE='/data/local/tools/z0/cache/flashinfer',CUDA_CACHE_PATH='/data/local/tools/z0/cache/cuda',SGLANG_ENABLE_JIT_DEEPGEMM='0',PATH='/data/local/envs/z0/bin:/usr/local/cuda/bin:'+os.environ.get('PATH',''),CUDA_HOME='/usr/local/cuda')
p=subprocess.Popen(['/data/local/envs/z0/bin/python','-u','-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=env,start_new_session=True)
receipt.write_text(json.dumps({'original_start_unix':original['started_unix'],'start_unix':start,'deadline_unix':deadline,'supervisor_pid':os.getpid(),'worker_pid':p.pid,'physical_gpu':4,'fixture_only':False,'training':False,'protocol':'directional_native_loop_v1','original_preparation_window_remains_closed':True,'write_limit_bytes':write_limit},indent=2))
reason=None;peak=0
try:
 while p.poll() is None:
  size=usage();peak=max(peak,size)
  if size>=write_limit or time.time()>=deadline:reason='storage/time guard';break
  time.sleep(5)
except BaseException as e:reason=repr(e)
if reason:
 os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
(state/f'{prefix}_EXIT.json').write_text(json.dumps({'exit_code':p.returncode,'reason':reason,'original_start_unix':original['started_unix'],'end_unix':time.time(),'peak_allocated_bytes':peak},indent=2))
