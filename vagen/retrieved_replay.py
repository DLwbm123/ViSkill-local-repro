"""Recover the audit using recorded tokens only; the generation server is never called."""
import json,time
from pathlib import Path
import torch
from verl import DataProto
from verl.workers.rollout.replica import TokenOutput
from . import retrieved_update_check as check
from .official_train_observer import write
from .agent_loop.gym_agent_loop_skill import GymAgentLoop
ROOT,SRC,OUT,P=check.ROOT,check.SRC,check.OUT,check.P
SOURCE=ROOT/'artifacts/reproduction'/P['replay_source']

class ReplayManager:
    def __init__(self,agent):
        self.agent=agent
        self.events=[json.loads(x) for x in (SOURCE/'requests/train_0002.jsonl').read_text().splitlines() if json.loads(x)['event']=='result' and json.loads(x)['seed']==agent.seed]
        self.events.sort(key=lambda x:x['number']);self.used=[]
    async def generate(self,**kwargs):
        event=self.events[len(self.used)]
        assert event['kind']==self.agent.kind
        assert kwargs['prompt_ids']==event['prompt_ids'] and kwargs['sampling_params']==event['sampling']
        assert len(kwargs.get('image_data') or [])==event['image_count']
        self.used.append(event['number'])
        return TokenOutput(token_ids=event['output_ids'])

class ReplayLoop(check.base.RunLoop):
    async def run(self,sampling_params,**kwargs):
        self.seed=int(kwargs['seed']);self.kind='rollout';manager=ReplayManager(self);self.server_manager=manager
        output=await GymAgentLoop.run(self,sampling_params,**kwargs)
        source=next(json.loads(x) for x in (SOURCE/'episodes/train_0002.jsonl').read_text().splitlines() if json.loads(x)['seed']==self.seed)
        assert output.response_ids==source['response_ids'] and output.response_mask==source['response_mask'] and output.num_turns==source['num_turns']
        for key in ['retrieved_skill_id','initial_env_state','trajectory_actions','trajectory_positions','distilled_strategy','traj_success','reward_extra_info']:
            assert check.base.plain(output.extra_fields[key])==source[key],(self.seed,key)
        assert len(manager.used)==len(manager.events)
        self.checks.append({'seed':self.seed,'matched_prompt_and_sampling_requests':len(manager.used),'source_request_numbers':manager.used,'response_ids_mask_trajectory_retrieval_match':True})
        return output

class SavedGeneration:
    def __init__(self,original):self.original=original;self.used=False
    def generate_sequences(self,prompts):
        assert not self.used;self.used=True
        expected=json.loads((OUT/'next_batch_manifest.json').read_text())['next_seeds']
        assert [int(x) for x in prompts.non_tensor_batch['seed']]==expected
        self.original.wake_up();self.original.sleep()
        saved=torch.load(OUT/'PRIVATE_REPLAY_GENERATION.pt',weights_only=False)
        data=DataProto.from_dict(**saved)
        write(OUT/'RECORDED_BATCH_USED.json',{'used_unix':time.time(),'episodes':128,'new_model_generation_calls':0,'source':P['replay_source'],'all_source_output_sequences_used':True})
        return data

class ReplayTrainer(check.CheckTrainer):
    def fit(self):
        self.async_rollout_manager=SavedGeneration(self.async_rollout_manager)
        return super().fit()
class ReplayTask(check.CheckTask):
    def run(self,config):
        import vagen.main_ppo_skill as entry
        import verl.utils.tracking as tracking
        torch.manual_seed(42);tracking.Tracking=check.CheckTracking;entry.RayPPOTrainer=ReplayTrainer
        return check.TaskRunner.run(self,config)
