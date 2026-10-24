"""Isolated solver fixtures only: never seed a training/cold-start library."""
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pytest
from vagen.skills.provider import ViSkillProvider
from vagen.skills.solver import FrozenLakeSolver

ROOT = Path(os.environ['TASK_ROOT'])
ART = ROOT / 'artifacts/component_tests/original_v1'
ART.mkdir(parents=True, exist_ok=True)


def save(name, data):
    (ART / name).write_text(json.dumps(data, indent=2))


@pytest.fixture(autouse=True)
def prohibit_external_distiller(monkeypatch):
    from vagen.skills.api_distiller import ViSkillDistiller
    def forbidden(*args, **kwargs):
        raise AssertionError('external distillation forbidden')
    monkeypatch.setattr(ViSkillDistiller, '_get_client', forbidden)


@pytest.fixture(scope='session')
def trajectory():
    state = {'desc': [list(s) for s in ['SFFF', 'FHHF', 'FFFF', 'FFFG']], 'player_pos': (0, 0)}
    data = FrozenLakeSolver().replay(state)
    assert data and len(data['actions']) == 6
    assert len(data['frames']) == len(data['positions']) == 7
    for i, frame in enumerate(data['frames']):
        frame.save(ART / f'original_{i}.png')
    save('fixture.json', {'source': 'deterministic BFS TEST FIXTURE ONLY; not policy rollout or cold-start',
         'state': state, 'actions': data['actions'], 'positions': data['positions']})
    return dict(data, state=state)


def provider_config(path):
    return {'library': {'path': str(path), 'max_size': 10},
            'cold_start': {'enable': False},
            'distill': {'mode': 'self', 'quality_ratio': 1, 'min_score': .9},
            'retrieval': {'similarity_weight': .9, 'min_score': .8},
            'utility': {'ema_alpha': .1, 'initial': .5},
            'render': {'strategy': {'enable': True, 'type': 'vision'},
                       'trajectory': {'enable': True, 'draw_path': True, 'thumbnail_size': 120}}}


@pytest.fixture
def provider(tmp_path):
    p = ViSkillProvider(provider_config(tmp_path / 'library'))
    p.init_env_type('FrozenLake')
    return p


@pytest.fixture(scope='session')
def strategy(trajectory):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
    from vagen.skills.distill_prompt import build_distill_prompt
    from vagen.skills.api_distiller import parse_strategy
    from vagen.skills.renderer import render_frames_with_path, FrozenLakeSkillRenderer
    start = time.time()
    set_seed(42)
    provider = ViSkillProvider(provider_config(ART / 'unused_library'))
    provider.init_env_type('FrozenLake')
    marked = render_frames_with_path(trajectory['frames'], trajectory['positions'], FrozenLakeSkillRenderer())
    for i, frame in enumerate(marked):
        frame.save(ART / f'annotated_{i}.png')
    trajectory_card = provider.render_trajectory(trajectory['frames'], trajectory['actions'], positions=trajectory['positions'])
    trajectory_card.save(ART / 'trajectory_only.png')
    prompt = build_distill_prompt(provider.task_description, trajectory['actions'], 'frozenlake', True)
    blocks = []
    for i, part in enumerate(prompt.split('<image>')):
        if part: blocks.append({'type': 'text', 'text': part})
        if i < 2: blocks.append({'type': 'image'})
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
    text = processor.apply_chat_template([{'role': 'user', 'content': blocks}], tokenize=False, add_generation_prompt=True) + '<strategy>'
    batch = processor(text=[text], images=[trajectory['frames'][0], trajectory_card], return_tensors='pt').to('cuda')
    length = batch.input_ids.shape[1]
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=256, do_sample=False)
    raw = '<strategy>' + processor.tokenizer.decode(out[0,length:], skip_special_tokens=True)
    parsed = parse_strategy(raw)
    card = provider.render_trajectory(trajectory['frames'], trajectory['actions'], positions=trajectory['positions'], strategy=parsed)
    card.save(ART / 'final_skill_card.png')
    card_tokens = processor(text=['<|vision_start|><|image_pad|><|vision_end|>'], images=[card], return_tensors='pt')
    result = {'raw_response': raw, 'parsed_strategy': parsed, 'strict_tags': bool(re.search(r'<strategy>.+?</strategy>',raw,re.S)),
              'model': 'pinned base current policy, no finetuning', 'fixture_only': True,
              'prompt_tokens': length, 'response_tokens': int(out.shape[1]-length),
              'card_size': list(card.size), 'card_image_grid_thw': card_tokens.image_grid_thw.tolist(),
              'card_visual_tokens': int(card_tokens.image_grid_thw.prod().item() // 4),
              'wall_seconds': time.time()-start, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    save('strategy.json', result)
    del model, batch, out, card_tokens
    torch.cuda.empty_cache()
    return dict(result, card=card)


@pytest.fixture
def skill_data(trajectory, strategy):
    from vagen.skills.features import FrozenLakeFeatureExtractor
    return {'features': FrozenLakeFeatureExtractor().extract(trajectory['state']),
            'strategy': strategy['parsed_strategy'], 'trajectory': trajectory['actions'],
            'image': strategy['card'], 'seed': 'test_fixture_only'}
