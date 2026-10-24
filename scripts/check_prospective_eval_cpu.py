"""CPU-only readiness of the actual serialized inference path."""
import os,json,runpy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from omegaconf import OmegaConf
from vagen.prospective_value_head_eval import validate_loading
from vagen.prospective_value_head_eval import EvalTrainer
from vagen.value_head_single_step import SingleTrainer
from verl import DataProto
from vagen.bootstrap_pilot import PilotDataset
from verl.utils.dataset.rl_dataset import collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.config import validate_config
from verl.utils import hf_tokenizer,hf_processor
from transformers import GenerationConfig
import torch,gymnasium as gym
from PIL import Image

ROOT=Path(os.environ['TASK_ROOT']);BASE=ROOT/'artifacts/component_tests/prospective_value_head_eval_v1'
make_config=runpy.run_path(str(ROOT/'src/scripts/run_prospective_value_head_eval.py'),run_name='config_check')['make_config']
m=json.loads((BASE/'data_manifest.json').read_text());cfgs={x:make_config(x,BASE/x) for x in ['b0','O','Z']}
for cfg in cfgs.values():validate_config(cfg,use_reference_policy=False,use_critic=False)
data=PilotDataset(str(ROOT/'src/configs/smoke/prospective_value_head_dev.yaml'),cfgs['b0'].data)
loader=StatefulDataLoader(data,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_fn)
assert [int(row['seed'][0]) for row in loader]==[row['seed'] for row in m['splits']['dev']]
fake=EvalTrainer.__new__(EvalTrainer);fake.global_steps=0;fake.async_rollout_mode=True
batch=DataProto.from_single_dict(collate_fn([data[0]]))
try:SingleTrainer._get_gen_batch(fake,batch)
except TypeError as exc:assert 'super(type, obj)' in str(exc)
else:raise AssertionError('old borrowed-super failure was not reproduced')
gen=fake._get_gen_batch(batch)
assert list(gen.non_tensor_batch['pilot_policy_version'])==[0] and list(gen.non_tensor_batch['fixed_context'])==['cards']
validate_loading({'missing_keys':['lm_head.weight']},True)
for info,tied in [({'unexpected_keys':['x']},True),({'mismatched_keys':['x']},True),({'error_msgs':['x']},True),({'missing_keys':['model.layers.0.weight']},True),({},False)]:
    try:validate_loading(info,tied)
    except AssertionError:pass
    else:raise AssertionError('invalid loading info accepted')
state=m['splits']['dev'][0]['state'];env=gym.make('FrozenLake-v1',desc=state['desc'],is_slippery=False,render_mode='rgb_array')
env.reset(seed=42);env.unwrapped.s=int(state['player_pos'][0])*4+int(state['player_pos'][1]);env.unwrapped.lastaction=None
initial=Image.fromarray(env.render()).convert('RGB');env.close()
card=Image.open(ROOT/'results/ppo_smoke/bootstrap_card_pair_v1/B/library/skill_4.png').convert('RGB')
msg=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':'Choose the next legal move on this board using the reference.'}]}]
reference=None;checks={}
for name,cfg in cfgs.items():
    tok=hf_tokenizer(cfg.actor_rollout_ref.model.path,trust_remote_code=True);proc=hf_processor(cfg.actor_rollout_ref.model.path,trust_remote_code=True)
    text=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);values=proc(text=[text],images=[card,initial],return_tensors='pt')
    meta={'template':proc.chat_template,'special':tok.special_tokens_map,'generation':GenerationConfig.from_pretrained(cfg.actor_rollout_ref.model.path).to_dict()}
    if reference is None:reference=(values,meta)
    else:
        assert meta==reference[1]
        for key in ['input_ids','attention_mask','pixel_values','image_grid_thw']:assert torch.equal(values[key],reference[0][key]),(name,key)
    checks[name]={'canonical_two_image_processor_exact_match':True,'common_generation_metadata':True,'rollout_dtype':str(cfg.actor_rollout_ref.rollout.dtype)}
result={'status':'CPU_PASS','dev_rows':len(data),'serial_loader_batches':len(loader),'order_exact':True,'loader_info_guard_valid_and_invalid_cases_checked':True,'processor_checks':checks,'model_forwards':0,'environment_steps':0,'generation':0,'runtime_HF_loader_and_actual_tie_checks':'Required during actual initialization before each policy generates'}
(BASE/'cpu_runner_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
