"""Freeze saved requests and reconstruct images, without model inference or env.step."""
import json, os, collections
from pathlib import Path
import numpy as np
from PIL import Image
import gymnasium as gym
from vagen.envs.frozenlake.utils.utils import parse_response

ROOT = Path(os.environ['TASK_ROOT'])
NAME = 'recorded_backend_audit_v2'
OLD = ROOT/'results/ppo_smoke/fixed_library_ppo_v1/B'
OUT = ROOT/f'artifacts/component_tests/{NAME}'

def render(state, position, previous_action):
    env = gym.make('FrozenLake-v1', desc=state['desc'], is_slippery=False, render_mode='rgb_array')
    try:
        env.reset(seed=42)
        env.unwrapped.s = int(position[0])*4+int(position[1])
        env.unwrapped.lastaction = previous_action
        return Image.fromarray(env.render()).convert('RGB')
    finally:
        env.close()

def main():
    assert not OUT.exists(), 'refuse replacing frozen inputs'
    OUT.mkdir(parents=True)
    requests = collections.defaultdict(list)
    for p in sorted((OLD/'requests').glob('*.json')):
        row=json.loads(p.read_text())
        if row['kind']=='rollout': requests[row['request_id']].append((p,row))
    episodes=[]
    for p in (OLD/'episodes').glob('*/episode.json'):
        row=json.loads(p.read_text()); row['_path']=str(p)
        episodes.append(row)
    selected=[]; skipped=[]
    for group in ['conditional','natural']:
        eligible=sorted([r for r in episodes if r['phase']=='dev' and r['policy_version']==0
            and r['fixed_context']=='cards' and r['fixed_group']==group and r['bfs']>1
            and (group!='natural' or r['retrieved_skill_id'] is None)],key=lambda r:r['seed'])
        picked=[]
        for row in eligible:
            if len(requests[row['mechanism_audit']['request_id']])<2:
                skipped.append({'seed':row['seed'],'reason':'fewer_than_two_rollout_requests'});continue
            picked.append(row)
            if len(picked)==2:break
        assert len(picked)==2
        selected.extend(picked)
    batch=json.loads((OLD/'updated_batches/1.json').read_text())
    training=[next(r for r in episodes if r['phase']=='train' and r['policy_version']==0
                   and r['seed']==seed) for seed in batch['seeds']]
    assert len(training)==32 and len(set(batch['seeds']))==32
    built={}; cases=[]
    lookup={'left':0,'down':1,'right':2,'up':3}
    for row in selected+training:
        key=row['mechanism_audit']['request_id']; saved=requests[key]
        assert saved and all(x[1]['seed']==row['seed'] for x in saved)
        folder=OUT/'images'/key;folder.mkdir(parents=True)
        source=Path(row['_path']).parent
        image_paths=[]
        if row['retrieved_skill_id'] is not None:
            original=source/'injected_skill.png'; assert original.exists()
            image_paths.append(str(original))
        initial=Image.open(source/'initial.png').convert('RGB')
        actual_initial=render(row['initial_env_state'],row['trajectory_positions'][0],None)
        assert np.array_equal(np.asarray(initial),np.asarray(actual_initial)), 'initial reconstruction mismatch'
        image_paths.append(str(source/'initial.png'))
        full=saved[-1][1]['prompt_ids']+saved[-1][1]['output_ids']
        prompt=saved[0][1]['prompt_ids']; assert full[:len(prompt)]==prompt
        response=full[len(prompt):]; mask=[0]*len(response)
        spans=[s for s in row['mechanism_audit']['spans'] if s['kind']=='generated']
        assert len(spans)==len(saved)
        consumed=0; previous=None; turns=[]
        for turn,((p,req),span) in enumerate(zip(saved,spans,strict=True)):
            offset=len(req['prompt_ids'])-len(prompt)
            assert offset==span['start'] and offset+len(req['output_ids'])==span['end']
            assert full[:len(req['prompt_ids'])]==req['prompt_ids']
            assert response[offset:span['end']]==req['output_ids']
            mask[offset:span['end']]=[1]*len(req['output_ids'])
            assert len(image_paths)==req['image_count']
            turns.append({'request_path':str(p),'start':offset,'end':span['end'],
                          'image_paths':list(image_paths),'raw':req['raw']})
            parsed=parse_response(response=req['raw'],action_sep=',',max_actions=5,prompt_format='free_think')
            for action in parsed.get('actions',[]):
                if action not in lookup:break
                if consumed>=len(row['trajectory_actions']):break
                assert action==row['trajectory_actions'][consumed]
                previous=lookup[action];consumed+=1
                pos=row['trajectory_positions'][consumed]
                if row['initial_env_state']['desc'][pos[0]][pos[1]] in ['G','H']:break
            if turn+1<len(saved):
                path=folder/f'observation_{turn+1}.png'
                render(row['initial_env_state'],row['trajectory_positions'][consumed],previous).save(path)
                image_paths.append(str(path))
        assert consumed==len(row['trajectory_actions'])
        terminal=render(row['initial_env_state'],row['trajectory_positions'][-1],previous)
        assert np.array_equal(np.asarray(Image.open(source/'terminal.png').convert('RGB')),np.asarray(terminal)), 'terminal reconstruction mismatch'
        assert mask==row['mechanism_audit']['response_mask']
        assert response==row['mechanism_audit']['response_ids']
        assert len(prompt)<=8192 and len(response)<=4096
        built[key]={'id':key,'seed':row['seed'],'category':row['category'],'phase':row['phase'],
            'group':row['fixed_group'],'source_episode':row['_path'],'prompt_ids':prompt,
            'response_ids':response,'response_mask':mask,'image_paths':image_paths,'turns':turns,
            'reward':row['reward_score'],'env_rewards':row['mechanism_audit']['env_rewards'],
            'success':row['traj_success'],'initial_and_terminal_pixel_match':True,
            'intermediate_image_identity':'reconstructed_from_saved_states_not_historical_tensor_identity'}
    for row in selected:
        key=row['mechanism_audit']['request_id']
        for turn in [0,1]:cases.append({'episode_id':key,'turn':turn,'request':built[key]['turns'][turn]['request_path']})
    manifest={'cases':cases,'score_episode_ids':[r['mechanism_audit']['request_id'] for r in selected],
        'training_episode_ids':[r['mechanism_audit']['request_id'] for r in training],
        'episodes':built,'selection_skips':skipped,'native_requests':8,
        'score_actor_rows':'4 full trajectories plus4 prefixes ending at second response; compares native prefix scores to true full-trajectory training layout and to prefix layout',
        'training_initial_head':str(OLD/'initialization/critic_rank0.json'),
        'original_resolved_config':str(OLD/'resolved.yaml'),'environment_steps':0,'model_calls':0}
    (OUT/'inputs.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'selected':[(r['seed'],r['fixed_group']) for r in selected],
        'score_requests':len(cases),'training_rows':len(training),'endpoint_checks':len(built)*2,
        'environment_steps':0,'model_calls':0,'inputs_bytes':(OUT/'inputs.json').stat().st_size}))

if __name__=='__main__':main()
