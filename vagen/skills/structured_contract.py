"""Diagnostic-only deterministic FrozenLake contract; not a visual perception oracle.

Applicability requires privileged environment state. Never infer correctness of a
policy's next action from correctness of the stored reference route.
"""
import json

MOVES = {'down': (1, 0), 'up': (-1, 0), 'left': (0, -1), 'right': (0, 1)}


def parse_contract(raw):
    contract = json.loads(raw)
    if not isinstance(contract, dict) or set(contract) != {'when', 'what'}:
        raise ValueError('contract keys')
    when, what = contract['when'], contract['what']
    if not isinstance(when, dict) or set(when) != {'goal_offset', 'requires_clear_route'}:
        raise ValueError('condition keys')
    offset = when['goal_offset']
    if not isinstance(offset, list) or len(offset) != 2 or any(type(v) is not int for v in offset):
        raise ValueError('integer offset required')
    if when['requires_clear_route'] is not True or not isinstance(what, list) or not what:
        raise ValueError('missing route condition/actions')
    actions = []
    for step in what:
        if not isinstance(step, dict) or set(step) != {'action', 'count'}:
            raise ValueError('action keys')
        if step['action'] not in MOVES or type(step['count']) is not int or not 1 <= step['count'] <= 64:
            raise ValueError('invalid action/count')
        actions.extend([step['action']] * step['count'])
    if len(actions) > 64:
        raise ValueError('oversized route')
    return contract, actions


def check_applicability(contract, actions, state):
    board = state['desc']
    row, col = map(int, state['player_pos'])
    goals = [(i, j) for i, line in enumerate(board) for j, tile in enumerate(line) if tile == 'G']
    if len(goals) != 1 or not board or any(len(line) != len(board[0]) for line in board):
        return False, 'invalid board'
    goal = goals[0]
    if [goal[0]-row, goal[1]-col] != contract['when']['goal_offset']:
        return False, 'goal offset mismatch'
    for index, action in enumerate(actions):
        dy, dx = MOVES[action];row += dy;col += dx
        if not 0 <= row < len(board) or not 0 <= col < len(board[0]):
            return False, 'boundary no-op'
        if board[row][col] == 'H':
            return False, 'hole on route'
        if board[row][col] == 'G' and index != len(actions)-1:
            return False, 'actions after terminal goal'
    return ((True, 'reference route reaches goal') if (row, col) == goal else (False, 'route misses goal'))


def validate_observed_contract(raw, row):
    contract, actions = parse_contract(raw)
    if not row.get('traj_success') or actions != row['trajectory_actions']:
        raise ValueError('not the observed successful actions')
    positions = row['trajectory_positions']
    if len(positions) != len(actions)+1:
        raise ValueError('missing observed positions')
    for before, after, action in zip(positions, positions[1:], actions):
        if tuple(int(b)-int(a) for a, b in zip(before, after)) != MOVES[action]:
            raise ValueError('recorded transition does not match deterministic action')
    offset = [int(b)-int(a) for a, b in zip(positions[0], positions[-1])]
    if contract['when']['goal_offset'] != offset:
        raise ValueError('wrong observed displacement')
    valid, reason = check_applicability(contract, actions, row['initial_env_state'])
    if not valid:
        raise ValueError(reason)
    return contract, actions
