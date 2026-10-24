"""One unfused actor reference on the already frozen eight input layouts."""
import json, os, time
from pathlib import Path
import torch
import ray
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from vagen.main_ppo_skill import TaskRunner
from vagen.recorded_backend_audit import CheckTrainer

ROOT=Path(os.environ['TASK_ROOT'])
OLD=ROOT/'artifacts/component_tests/recorded_backend_audit_v2'
OUT=ROOT/'artifacts/component_tests/kernel_score_reference_v1'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2))

def compare_prefix(batch):
    checks=[]
    for i in range(4):
        a=batch.batch['attention_mask'][i].bool();b=batch.batch['attention_mask'][i+4].bool()
        n=int(b.sum())
        assert torch.equal(batch.batch['input_ids'][i,a][:n],batch.batch['input_ids'][i+4,b])
        assert torch.equal(batch.batch['position_ids'][i,:,a][:,:n],batch.batch['position_ids'][i+4,:,b])
        full=batch.non_tensor_batch['multi_modal_inputs'][i];prefix=batch.non_tensor_batch['multi_modal_inputs'][i+4]
        for key in ['image_grid_thw','pixel_values']:
            assert torch.equal(full[key][:len(prefix[key])],prefix[key]),key
        checks.append({'episode_index':i,'valid_prefix_tokens':n,'ids_positions_pixels_exact_prefix':True})
    return checks

class ReferenceWorker(ActorRolloutRefWorker):
    def __init__(self,config,role):
        super().__init__(config,role='actor')
        assert not self._is_rollout and not self.config.model.use_fused_kernels

class ReferenceTrainer(CheckTrainer):
    def fit(self):
        assert not self.use_critic and not self.async_rollout_mode
        manifest=json.loads((OLD/'inputs.json').read_text());episodes=manifest['episodes']
        full=[episodes[k] for k in manifest['score_episode_ids']];prefix=[]
        for row in full:
            cut=row['turns'][1]['end']
            prefix.append(dict(row,response_ids=row['response_ids'][:cut],response_mask=row['response_mask'][:cut],image_paths=row['turns'][1]['image_paths']))
        batch=self.make_batch(full+prefix)
        save('prefix_identity.json',{'checks':compare_prefix(batch),'scope':'Current reconstructed full versus prefix layout; no historical intermediate tensor claim'})
        assert not (OUT/'actor_started.json').exists(),'refuse replaying completed or dispatched samples'
        save('actor_started.json',{'samples':8,'started_unix':time.time(),'generated_tokens':0,'optimizer_steps':0})
        result=self.actor_rollout_wg.compute_log_prob(batch).batch['old_log_probs'].cpu()
        assert torch.isfinite(result).all()
        torch.save(result,OUT/'actor.pt')
        previous=torch.load(OLD/'score_actor.pt',map_location='cpu',weights_only=True)['log_probs']
        rows=[]
        for i,case in enumerate(manifest['cases']):
            idx=manifest['score_episode_ids'].index(case['episode_id']);turn=episodes[case['episode_id']]['turns'][case['turn']]
            req=json.loads(Path(case['request']).read_text());n=len(req['output_ids'])
            native=json.loads((OLD/f'native/{i}.json').read_text())['meta_info']['input_token_logprobs'][-n:]
            assert [int(x[1]) for x in native]==req['output_ids']
            target=torch.tensor([x[0] for x in native]);a=result[idx,turn['start']:turn['end']].float()
            d=(a-target).abs();old=(previous[idx,turn['start']:turn['end']].float()-target).abs()
            row={'case':i,'seed':episodes[case['episode_id']]['seed'],'turn':case['turn']+1,
                 'max_abs_nats':float(d.max()),'mean_abs_nats':float(d.mean()),
                 'old_fused_max_abs_nats':float(old.max()),'old_fused_mean_abs_nats':float(old.mean()),
                 'gate_pass':float(d.max())<=.05 and float(d.mean())<=.005,
                 'log_probs':a.tolist(),'native_log_probs':target.tolist(),'token_ids':req['output_ids']}
            if case['turn']==1:
                p=result[idx+4,turn['start']:turn['end']].float()
                row.update(prefix_max_abs_nats=float((p-target).abs().max()),prefix_mean_abs_nats=float((p-target).abs().mean()),full_vs_prefix_max_abs_nats=float((p-a).abs().max()))
            rows.append(row)
        save('results.json',{'cases':rows,'all_eight_probability_gate_pass':all(r['gate_pass'] for r in rows),
            'scope':'Only unfused actor reference; same saved native scores and reconstructed inputs. No full backend or historical equivalence proof.'})
        save('COMPLETE.json',{'completed_unix':time.time(),'actor_forward_samples':8,'native_requests':0,'generated_tokens':0,'environment_steps':0,'optimizer_steps':0})

class ReferenceTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self,config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(ReferenceWorker)
        return ReferenceWorker,RayWorkerGroup
    def run(self,config):
        import vagen.main_ppo_skill as entry
        entry.RayPPOTrainer=ReferenceTrainer
        return super().run(config)
