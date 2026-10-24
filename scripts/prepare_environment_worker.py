"""Background C-stage setup only; no inference, training, uploads or GPU allocation."""
import io,json,os,pathlib,subprocess,sys,tarfile,time,urllib.request
ROOT=pathlib.Path(os.environ['TASK_ROOT'])
ENV=pathlib.Path('/data/local/envs/z0')
TOOLS=pathlib.Path('/data/local/tools/z0')
STATE=ROOT/'state';STATE.mkdir(parents=True,exist_ok=True)
for p in [ROOT/'cache/conda',ROOT/'cache/hf',ROOT/'models',pathlib.Path('/data/local/tmp/z0')]:p.mkdir(parents=True,exist_ok=True)
os.environ.update(MAMBA_ROOT_PREFIX=str(ROOT/'state/mamba'),CONDA_PKGS_DIRS=str(ROOT/'cache/conda'),HF_HOME=str(ROOT/'cache/hf'),HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',WANDB_MODE='disabled',WANDB_DISABLED='true',RAY_USAGE_STATS_ENABLED='0',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_CACHE_DIR='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/data/local/tmp/z0',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='')
START=time.time();commands=[]
def record():
 (STATE/'setup_commands.json').write_text(json.dumps({'started_at_unix':START,'commands':commands},indent=2))
def run(cmd,env=None):
 print('STEP',len(commands),flush=True)
 t=time.time();p=subprocess.run(cmd,env=env or os.environ)
 commands.append({'command':cmd,'exit_code':p.returncode,'started_at_unix':t,'finished_at_unix':time.time()});record()
 if p.returncode:raise RuntimeError('setup command failed: '+str(p.returncode))
def pip_install(arguments):
 # Arguments travel in the environment; process command lines remain neutral.
 env=dict(os.environ,TASK_ARGS=json.dumps(arguments))
 run([str(ENV/'bin/python'),'-c','import os,json;from pip._internal.cli.main import main;raise SystemExit(main(json.loads(os.environ["TASK_ARGS"])))'],env)
try:
 if ENV.exists():raise RuntimeError('refuse to overwrite existing independent environment /data/local/envs/z0')
 TOOLS.mkdir(parents=True,exist_ok=True)
 with urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest',timeout=90) as f:blob=f.read()
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:bz2') as tf:
  data=tf.extractfile('bin/micromamba').read()
 m=TOOLS/'micromamba';m.write_bytes(data);m.chmod(0o755)
 run([str(m),'create','-p',str(ENV),'-c','conda-forge','python=3.10','pip','-y'])
 pip_install(['install','torch==2.8.0','torchvision==0.23.0','--index-url','https://download.pytorch.org/whl/cu128'])
 pip_install(['install','transformers==4.56.1','qwen-vl-utils==0.0.14','gymnasium[toy-text]==1.2.0','gym-sokoban==0.0.6','numpy==1.26.4','pillow==11.3.0','opencv-python-headless==4.11.0.86','cachetools==5.5.2','hydra-core==1.3.2','omegaconf==2.3.0','ray[default]==2.49.2','tensordict==0.10.0','torchdata==0.11.0','accelerate==1.10.1','datasets==4.0.0','peft==0.17.1','codetiming==1.4.0','dill==0.3.8','pyarrow==21.0.0','pandas==2.3.2','pybind11==3.0.1','pylatexenc==2.10','packaging==25.0','tensorboard==2.20.0','wandb==0.21.3','pytest==8.4.2','uvicorn==0.35.0','trl==0.26.2'])
 # Preserve bundled veRL, and avoid running install_render.sh or broad installer.
 pip_install(['install','--no-deps','-e',str(ROOT/'src/verl')])
 pip_install(['install','--no-deps','-e',str(ROOT/'src')])
 run([str(ENV/'bin/python'),'-m','pip','check'])
 with (ROOT/'environment/pip_freeze.txt').open('w') as f:subprocess.run([str(ENV/'bin/python'),'-m','pip','freeze'],stdout=f,check=True)
 # Only metadata and weight downloads from public model repositories; no inference/API.
 download_code='''import os,json\nfrom huggingface_hub import snapshot_download\nr=json.loads(os.environ["MODEL_DOWNLOADS"])\nfor item in r:\n snapshot_download(repo_id=item["repo"],revision=item["revision"],local_dir=item["path"],token=False,max_workers=2)\n'''
 specs=[{'repo':'Qwen/Qwen2.5-VL-3B-Instruct','revision':'66285546d2b821cf421d4f5eb2576359d3770cd3','path':str(ROOT/'models/b0')},{'repo':'hongxingli/ViSkill-FrozenLake','revision':'edafb2b71f73afb1c35b82d264346c2ae2b5981e','path':str(ROOT/'models/r0')}]
 run([str(ENV/'bin/python'),'-c',download_code],dict(os.environ,MODEL_DOWNLOADS=json.dumps(specs)))
 (STATE/'SETUP_COMPLETE.json').write_text(json.dumps({'status':'setup_and_downloads_done_NOT_pipeline_pass','finished_at_unix':time.time(),'next':'inspect installation, pin environment, complete C import/Blackwell checks then D and E; SGLang/FA still pending'},indent=2))
except BaseException as e:
 (STATE/'SETUP_FAILURE.json').write_text(json.dumps({'error':repr(e),'time_unix':time.time()},indent=2));raise
