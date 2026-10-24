"""Posthoc action/termination diagnosis of all fixed episodes; no model calls."""
import json,os
from collections import Counter,deque
from pathlib import Path
ROOT=Path(os.environ.get('TASK_ROOT',str(Path(__file__).resolve().parents[1])))/'artifacts/component_tests/skill_capacity_v1'
def distance(board,position):
 position=tuple(position)
 if board[position[0]][position[1]]=='H':return None
 q=deque([(position,0)]);seen={position}
 while q:
  (r,c),d=q.popleft()
  if board[r][c]=='G':return d
  for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
   rr,cc=r+dr,c+dc
   if 0<=rr<len(board) and 0<=cc<len(board[0]) and board[rr][cc]!='H' and (rr,cc) not in seen:
    seen.add((rr,cc));q.append(((rr,cc),d+1))
 return None
rows=[];failure=Counter();first=Counter();invalid_total=0;noop_total=0
for seed in range(20001,20129):
 row=json.loads((ROOT/f'episode_{seed}/episode.json').read_text());a=row['mechanism_audit'];board=row['initial_env_state']['desc']
 positions=row['trajectory_positions'];actions=row['trajectory_actions'];assert len(positions)==len(actions)+1
 initial=distance(board,positions[0]);last=positions[-1];on_hole=board[last[0]][last[1]]=='H'
 turns=sum(s['kind']=='generated' for s in a['spans']);invalid_turns=sum(v==0 for v in a['env_rewards'])
 noops=sum(tuple(x)==tuple(y) for x,y in zip(positions,positions[1:]));noop_total+=noops;invalid_total+=invalid_turns
 if not actions:category='no_valid_action'
 elif on_hole:category='hole'
 elif turns>=5:category='turn_limit'
 else:category='other_stop'
 if not row['traj_success']:failure[category]+=1
 next_distance=distance(board,positions[1]) if actions else None
 first_type=('no_valid_action' if not actions else 'hole_or_unreachable' if next_distance is None else
  'reduces_shortest_distance' if next_distance==initial-1 else 'boundary_noop' if positions[0]==positions[1] else 'nonprogress_move')
 first[first_type]+=1
 rows.append({'seed':seed,'success':row['traj_success'],'bfs':initial,'actions':len(actions),'turns':turns,
  'zero_reward_turns':invalid_turns,'boundary_noops':noops,'failure_category':category if not row['traj_success'] else None,'first_action_class':first_type})
print(json.dumps({'scope':'posthoc all128; shortest-distance classification uses verifier state, never model input',
 'episodes':128,'failed':sum(not r['success'] for r in rows),'failure_categories':dict(failure),
 'first_valid_action_class_all128':dict(first),'zero_reward_turns':invalid_total,'boundary_noop_actions':noop_total,
 'episodes_with_zero_reward_turn':sum(r['zero_reward_turns']>0 for r in rows),
 'failed_episodes_with_noop':sum(not r['success'] and r['boundary_noops']>0 for r in rows),
 'rows':rows},indent=2))
