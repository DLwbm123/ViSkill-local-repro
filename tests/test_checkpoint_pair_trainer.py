"""Exercise actual trainer checkpoint methods with CPU dummy workers, never PPO."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from vagen.ray_trainer_skill import RayPPOTrainerSkill
from vagen.skills.provider import ViSkillProvider

class Worker:
    def __init__(self,value):self.value=torch.tensor([value]);self.load_calls=0
    def save_checkpoint(self,local,remote,step,max_ckpt_to_keep=None):
        Path(local).mkdir(parents=True,exist_ok=True);torch.save(self.value,Path(local)/'dummy_weight.pt')
    def load_checkpoint(self,local,del_local_after_load=False):
        self.load_calls+=1;self.value=torch.load(Path(local)/'dummy_weight.pt',weights_only=True)

class Loader:
    def state_dict(self):return {'cursor':2}
    def load_state_dict(self,state):self.restored=state

def trainer(tmp_path):
    obj=object.__new__(RayPPOTrainerSkill)
    obj.config=OmegaConf.create({'trainer':{'default_local_dir':str(tmp_path/'checkpoints'),'default_hdfs_dir':None,'resume_mode':'resume_path','resume_from_path':str(tmp_path/'checkpoints/global_step_2'),'del_local_ckpt_after_load':False},'skill_system':{'enable':True,'checkpoint_pairing':{'enable':True},'library':{'path':str(tmp_path/'active')}}})
    obj.global_steps=2;obj.use_critic=True;obj.actor_rollout_wg=Worker(2.);obj.critic_wg=Worker(7.);obj.train_dataloader=Loader()
    provider=ViSkillProvider({'library':{'path':str(tmp_path/'active')}});provider.init_env_type('FrozenLake')
    provider.library.admit({'features':[0.]*10,'strategy':'dummy component text','trajectory':['right'],'positions':[[0,0],[0,1]],'image':Image.new('RGB',(64,64),'blue')});provider.update_utility('skill_0',1.);provider.save()
    obj._distill_callback=SimpleNamespace(provider=provider)
    return obj

def test_actual_trainer_methods_restore_dummy_weights_and_library(tmp_path):
    obj=trainer(tmp_path);obj._save_checkpoint()
    assert (tmp_path/'checkpoints/latest_checkpointed_iteration.txt').read_text()=='2'
    obj.actor_rollout_wg.value.fill_(-1);obj.critic_wg.value.fill_(-1)
    obj._distill_callback.provider.update_utility('skill_0',0.);obj._distill_callback.provider.save()
    obj._load_checkpoint()
    assert obj.actor_rollout_wg.value.item()==2. and obj.critic_wg.value.item()==7.
    assert obj._distill_callback.provider.library._skills[0]['utility']==pytest.approx(.55)
    assert obj.train_dataloader.restored=={'cursor':2}

def test_validation_runs_before_any_worker_load(tmp_path):
    obj=trainer(tmp_path);obj._save_checkpoint()
    (tmp_path/'checkpoints/global_step_2/paired_skill_library/skill_0.png').unlink()
    with pytest.raises(ValueError,match='image'):obj._load_checkpoint()
    assert obj.actor_rollout_wg.load_calls==obj.critic_wg.load_calls==0

def test_failed_sidecar_does_not_advance_latest_tracker(tmp_path):
    obj=trainer(tmp_path);obj._save_checkpoint();obj.global_steps=3
    (tmp_path/'active/skill_0.png').unlink()
    with pytest.raises(ValueError,match='image'):obj._save_checkpoint()
    assert (tmp_path/'checkpoints/latest_checkpointed_iteration.txt').read_text()=='2'
