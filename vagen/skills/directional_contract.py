"""Diagnostic representation ablation. Environment facts only reject, never edit text."""
import json
from .structured_contract import parse_contract, validate_observed_contract, check_applicability

DIRECTIONS = {'up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right'}


def build_directional_prompt(actions):
    return ('Summarize this actual successful FrozenLake trajectory as a reusable JSON contract. '
            'The two images show its initial and terminal states. The executed actions below reached the goal. '
            'Use ONLY keys when and what. when has goal_direction (up/down/left/right/up_left/up_right/down_left/down_right relative to the player), goal_manhattan_distance (positive integer from start to goal), '
            'and requires_clear_route (boolean true, a REQUIRED applicability condition). '
            'Rows increase downward; columns increase rightward. what is a list of action/count objects; '
            'action is up/down/left/right and count is a positive integer. Preserve the observed action sequence. '
            'Do not add corner names, explanations, markdown or other keys. Return only JSON.\n'
            'Observed terminal success: true\nExecuted actions: '+json.dumps(actions))


def require_deterministic_state(state):
    # Unknown dynamics are not assumed safe. Old saved records need an explicit
    # provenance-backed migration before using this optional runtime guard.
    if state.get('is_slippery') is not False or state.get('slip_prob') != 0:
        raise ValueError('directional contract requires explicit deterministic dynamics')


def displacement_condition(dy, dx):
    vertical = 'up' if dy < 0 else 'down' if dy > 0 else ''
    horizontal = 'left' if dx < 0 else 'right' if dx > 0 else ''
    return '_'.join(x for x in [vertical, horizontal] if x), abs(dy)+abs(dx)


def parse_directional_contract(raw, *, allow_assistant_envelope=False):
    body = raw.strip()
    if allow_assistant_envelope and body.startswith('assistant\n'):
        body = body[len('assistant\n'):].strip()
    if body.startswith('```json\n') and body.endswith('\n```'):
        body = body[len('```json\n'):-len('\n```')]
    contract = json.loads(body)
    if not isinstance(contract, dict) or set(contract) != {'when', 'what'}:
        raise ValueError('contract keys')
    when = contract['when']
    if not isinstance(when, dict) or set(when) != {'goal_direction', 'goal_manhattan_distance', 'requires_clear_route'}:
        raise ValueError('condition keys')
    if when['goal_direction'] not in DIRECTIONS:
        raise ValueError('invalid relative direction')
    distance = when['goal_manhattan_distance']
    if type(distance) is not int or distance <= 0:
        raise ValueError('positive integer distance required')
    # Reuse the action/type validator. This placeholder never leaves validation
    # and is never treated as a predicted offset or saved in a skill.
    proxy = {'when': {'goal_offset': [0, 0], 'requires_clear_route': when['requires_clear_route']},
             'what': contract['what']}
    _, actions = parse_contract(json.dumps(proxy))
    return contract, actions


def _matching_proxy(contract, dy, dx):
    direction, distance = displacement_condition(dy, dx)
    when = contract['when']
    if direction != when['goal_direction'] or distance != when['goal_manhattan_distance']:
        raise ValueError('relative goal condition mismatch')
    return {'when': {'goal_offset': [int(dy), int(dx)], 'requires_clear_route': when['requires_clear_route']},
            'what': contract['what']}


def validate_directional_observation(raw, row, *, allow_assistant_envelope=False):
    contract, actions = parse_directional_contract(raw, allow_assistant_envelope=allow_assistant_envelope)
    positions = row['trajectory_positions']
    dy, dx = [int(b)-int(a) for a, b in zip(positions[0], positions[-1])]
    proxy = _matching_proxy(contract, dy, dx)
    validate_observed_contract(json.dumps(proxy), row)
    return contract, actions


def check_directional_applicability(contract, actions, state):
    goals = [(i, j) for i, line in enumerate(state['desc']) for j, tile in enumerate(line) if tile == 'G']
    if len(goals) != 1:
        return False, 'invalid goal count'
    dy, dx = [int(g)-int(p) for g, p in zip(goals[0], state['player_pos'])]
    try:
        proxy = _matching_proxy(contract, dy, dx)
    except ValueError as exc:
        return False, str(exc)
    return check_applicability(proxy, actions, state)
