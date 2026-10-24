"""Offline Transformers evaluation using the official FrozenLake and skill APIs.

This diagnostic backend retokenizes complete chat history each turn. It does not
claim identity with the trainer's incremental token concatenation or perform PPO.
"""
import asyncio
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

from vagen.envs.frozenlake.frozenlake_env import FrozenLake
from vagen.envs.frozenlake.utils.utils import parse_response
from vagen.skills.provider import ViSkillProvider
from vagen.skills.reward import compute_reward, compute_prefix_reward

ROOT = Path(os.environ["TASK_ROOT"])
ARM = os.environ["TASK_ARM"]
CFG = json.loads((ROOT / "src/configs/low_resource/local_eval.json").read_text())
OUT = ROOT / "results/local_eval"
ART = ROOT / "artifacts/local_eval" / ARM
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def content(obs):
    """Same image-placeholder order as official convert_obs_to_content."""
    images = (obs.get("multi_modal_input") or {}).get("<image>", [])
    segments = obs["obs_str"].split("<image>")
    assert len(segments) - 1 == len(images), "image/placeholder mismatch"
    blocks = []
    for i, segment in enumerate(segments):
        if segment:
            blocks.append({"type": "text", "text": segment})
        if i < len(images):
            blocks.append({"type": "image"})
    return blocks, list(images)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial))


async def main():
    results_path = OUT / f"{ARM}.jsonl"
    if results_path.exists():
        raise RuntimeError("refuse to overwrite or silently rerun an existing arm")
    start = time.time()
    set_seed(CFG["generation_seed"])
    spec = CFG["arms"][ARM]
    model_path = ROOT / "models" / spec["model"]
    print("LOAD", ARM, flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, device_map={"": "cuda:0"},
    ).eval()
    torch.cuda.reset_peak_memory_stats()
    provider = None
    if spec["skills"]:
        provider = ViSkillProvider({
            "library": {"path": str(model_path / "skill_library"), "max_size": 10000},
            "retrieval": {"similarity_weight": CFG["similarity_weight"], "min_score": CFG["retrieval_threshold"]},
            "distill": {"mode": "self"}, "cold_start": {"enable": False},
            "render": {"strategy": {"enable": True, "type": "vision"}},
        })
        provider.init_env_type("FrozenLake")
        assert len(provider.library) == 265, "released library size differs from pinned audit"
        # Missing card is an error, never silently fall back to text-only skills.
        for skill in provider.library._skills:
            assert len(skill["features"]) == 10 and skill["trajectory"] and skill["strategy"]
            assert "utility" in skill and "use_count" in skill
            image = provider.library.load_image(skill, provider.library_path)
            assert image is not None
            image.close()
    rows = []
    for seed in CFG["seeds"]:
        ep_start = time.time()
        epdir = ART / str(seed)
        epdir.mkdir(exist_ok=False)
        env = FrozenLake(CFG["environment"])
        turns, rewards, actions = [], [], []
        skill = None
        candidate_score = None
        try:
            obs, _ = await env.reset(seed=seed)
            initial = env.get_state_for_skill()
            obs["multi_modal_input"]["<image>"][0].save(epdir / "initial.png")
            blocks, images = content(await env.system_prompt())
            messages = [{"role": "system", "content": blocks}]
            if provider:
                candidate = provider.retrieve(initial)
                candidate_score = candidate["score"] if candidate else None
                if candidate and candidate_score >= CFG["retrieval_threshold"]:
                    skill = candidate
                    skill_obs = provider.format_for_prompt(skill)
                    sb, si = content(skill_obs)
                    assert si, "visual skill injection required"
                    messages[0]["content"].extend(sb)
                    images.extend(si)
                    si[0].save(epdir / "retrieved_skill.png")
            success, done = False, False
            for turn in range(CFG["max_turns"]):
                blocks, current_images = content(obs)
                images.extend(current_images)
                messages.append({"role": "user", "content": blocks})
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                batch = processor(text=[text], images=images, return_tensors="pt").to("cuda")
                n_prompt = batch.input_ids.shape[1]
                if n_prompt + CFG["max_new_tokens"] > CFG["max_context_tokens"]:
                    raise RuntimeError(f"context budget exceeded: {n_prompt}; no silent truncation")
                tick = time.time()
                with torch.inference_mode():
                    generated = model.generate(**batch, max_new_tokens=CFG["max_new_tokens"], do_sample=False,
                                               pad_token_id=processor.tokenizer.pad_token_id)
                tokens = generated[0, n_prompt:]
                response = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                elapsed = time.time() - tick
                grid = batch.image_grid_thw.detach().cpu().tolist()
                del batch, generated
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
                obs, reward, done, info = await env.step(response)
                obs["multi_modal_input"]["<image>"][0].save(epdir / f"turn_{turn+1}.png")
                rewards.append(float(reward))
                actions.extend(info["actions"])
                success = bool(info["success"])
                record = {
                    "turn": turn, "model_response": response, "parsed_actions": info["actions"],
                    # The environment replaces info['actions'] with actions actually executed.
                    # Preserve that legacy field while recording the parser output separately.
                    "parser_actions": parse_response(response, prompt_format="free_think", max_actions=5)["actions"],
                    "executed_actions": info["actions"],
                    "format_correct": bool(info.get("format_correct", False)),
                    "action_valid": bool(info["metrics"]["turn_metrics"]["action_is_valid"]),
                    "environment_reward": float(reward), "done": done, "success": success,
                    "state": env.get_state_for_skill(), "prompt_tokens": n_prompt,
                    "response_tokens": len(tokens), "image_grid_thw": grid,
                    "image_count": len(images), "generation_seconds": elapsed,
                }
                turns.append(record)
                dump(epdir / f"turn_{turn+1}.json", record)
                print("TURN", ARM, seed, turn, "actions", info["actions"], "success", success, flush=True)
                if done:
                    break
            row = {
                "arm": ARM, "seed": seed, "initial_state": initial,
                "initial_observation": str(epdir / "initial.png"),
                "retrieved_skill_id": skill["id"] if skill else None,
                "retrieval_score": skill["score"] if skill else None,
                "candidate_retrieval_score": candidate_score,
                "skill_card_path": str(epdir / "retrieved_skill.png") if skill else None,
                "turns": turns, "actions": actions, "success": success,
                "terminal_state": env.get_state_for_skill(),
                "terminated": done, "truncated_at_turn_limit": not done,
                "episode_turns": len(turns), "primitive_actions": len(actions),
                "reward_breakdown": {
                    "environment_total": sum(rewards), "outcome": float(success),
                    "format_total": sum(rewards)-float(success),
                    "official_prefix": compute_prefix_reward(skill["trajectory"], actions) if skill and not success else 0.,
                    "total": compute_reward(rewards, "skill_guided" if skill else "success", success,
                                            skill["trajectory"] if skill else None, actions, CFG["guided_weight"]),
                },
                "wall_seconds": time.time()-ep_start,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            with results_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=serial)+"\n")
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(row)
        finally:
            await env.close()
    turns = [t for row in rows for t in row["turns"]]
    scores = [row["retrieval_score"] for row in rows if row["retrieval_score"] is not None]
    summary = {
        "arm": ARM, "episodes": len(rows), "success_rate": sum(r["success"] for r in rows)/len(rows),
        "mean_episode_turns": float(np.mean([r["episode_turns"] for r in rows])),
        "mean_primitive_actions": float(np.mean([r["primitive_actions"] for r in rows])),
        "invalid_action_turn_rate": sum(not t["action_valid"] for t in turns)/len(turns),
        "skill_retrieval_fraction": len(scores)/len(rows),
        "mean_retrieval_score": float(np.mean(scores)) if scores else None,
        "mean_prompt_tokens": float(np.mean([t["prompt_tokens"] for t in turns])),
        "max_prompt_tokens": max(t["prompt_tokens"] for t in turns),
        "mean_response_tokens": float(np.mean([t["response_tokens"] for t in turns])),
        "wall_seconds": time.time()-start,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "all_episodes_finished": len(rows) == len(CFG["seeds"]),
        "has_valid_action": any(t["action_valid"] for t in turns),
        "training": False, "protocol": CFG,
    }
    dump(OUT / f"{ARM}_summary.json", summary)
    print("ARM_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
