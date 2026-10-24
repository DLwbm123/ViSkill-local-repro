"""Reward computation for skill-guided training."""

from typing import List, Optional


def compute_state_progress_reward(reference_positions, model_positions, *, bounded=True):
    """FrozenLake relative-state Eq.6, optionally capped to satisfy its stated bound.

    Positions include the initial state and one observation per executed action.
    The upper cap is an explicit engineering variant: Eq.6 itself can exceed one
    when a shorter path reaches a late reference state in a different layout.
    """
    if not reference_positions or len(reference_positions) < 2:
        return 0.0
    if not model_positions:
        raise ValueError("state progress requires observed model positions")

    def relative(positions):
        origin = positions[0]
        if any(len(p) != 2 for p in positions):
            raise ValueError("FrozenLake positions must be row/column pairs")
        return [(int(p[0])-int(origin[0]), int(p[1])-int(origin[1])) for p in positions]

    reference = relative(reference_positions)
    visited = set(relative(model_positions))
    furthest = max(i for i, position in enumerate(reference) if position in visited)
    model_length = len(model_positions)-1
    value = max(0.0, (2*furthest-model_length)/(len(reference)-1))
    return min(1.0, value) if bounded else value


def compute_prefix_reward(skill_trajectory: List[str], model_trajectory: List[str]) -> float:
    """Reward based on how closely model followed skill's trajectory (prefix matching)."""
    if not skill_trajectory:
        return 0.0
    prefix_len = 0
    for s_act, m_act in zip(skill_trajectory, model_trajectory):
        if s_act == m_act:
            prefix_len += 1
        else:
            break
    extra_steps = max(0, len(model_trajectory) - prefix_len)
    return max(0.0, (prefix_len - extra_steps) / len(skill_trajectory))


def compute_reward(
    env_rewards: List[float],
    mode: str,
    traj_success: bool,
    skill_trajectory: Optional[List[str]] = None,
    model_trajectory: Optional[List[str]] = None,
    guided_weight: float = 0.5,
    reference_positions=None,
    model_positions=None,
) -> float:
    """Compute final reward based on mode.

    env_rewards already includes per-step signals from the environment:
      - format_reward (valid action bonus)
      - success_reward (task completion)
      - process_reward (BFS distance change, only when mode=bfs_process)

    This function adds mode-specific bonuses on top:
      success:      env_rewards only
      bfs_process:  env_rewards only (process reward already accumulated per-step)
      skill_guided: env_rewards + weighted prefix match bonus on failure
    """
    base = sum(env_rewards) if env_rewards else 0.0

    if mode == "state_progress_bounded" and not traj_success and skill_trajectory:
        if reference_positions is None or len(reference_positions) != len(skill_trajectory)+1:
            raise ValueError("reference position/action lengths differ; no action-only fallback")
        if model_positions is None or len(model_positions) != len(model_trajectory or [])+1:
            raise ValueError("observed model position/action lengths differ")
        return base + guided_weight * compute_state_progress_reward(reference_positions, model_positions)

    if mode == "skill_guided" and not traj_success:
        if skill_trajectory and model_trajectory:
            return base + guided_weight * compute_prefix_reward(skill_trajectory, model_trajectory)

    return base
