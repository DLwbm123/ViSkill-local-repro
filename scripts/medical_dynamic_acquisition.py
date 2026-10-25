"""Current-case categorical tool use, with visual-source and wrong-case controls."""
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
NAME = 'dynamic_acquisition_tools_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
PARENT = ROOT / 'artifacts/medical/continual_acquisition_bank_v1'
COHORT = ROOT / 'artifacts/medical/acquisition_to_organ_transfer_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from slake_probe import normalize, query_view
read, write = common.read, common.write
QUERIES = {
    'modality': 'What is the imaging modality of this image? Output only CT, MRI, X-Ray, T1, T2, or Unknown.',
    'plane': 'What is the imaging plane of this image? Output only Transverse, Coronal, Sagittal, or Unknown.'}
ALIASES = {
    'modality': {'ct': 'CT', 'mri': 'MRI', 'x ray': 'X-Ray', 'xray': 'X-Ray', 't1': 'T1', 't1 weighted': 'T1', 't1 weighted mri': 'T1', 't2': 'T2', 't2 weighted': 'T2', 't2 weighted mri': 'T2'},
    'plane': {'transverse': 'Transverse', 'transverse plane': 'Transverse', 'axial': 'Transverse', 'axial plane': 'Transverse', 'coronal': 'Coronal', 'coronal plane': 'Coronal', 'sagittal': 'Sagittal', 'sagittal plane': 'Sagittal'}}


def parse(kind, raw):
    return ALIASES[kind].get(normalize(raw), 'Unknown')


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def reader_query(row, tags):
    assert len(tags) == 2 and all(t in set(ALIASES[k].values()) | {'Unknown'} for k, t in zip(QUERIES, tags))
    q = query_view(row)
    q['question'] = ('A separate acquisition tool returned categorical predictions, which may be incorrect.\n'
                     f'Modality prediction: {tags[0]}.\nPlane prediction: {tags[1]}.\n'
                     'Use the current image to answer this question:\n' + q['question'])
    assert set(q) == {'qid', 'img_name', 'question'}
    return q


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    assert read(PARENT / 'RESULT.json')['object_gate_pass']['external_module_bank']
    m = read(COHORT / 'private_manifest.json')
    primary, background = m['new_validation'], m['old_background']
    rows = primary + background
    flags = read(COHORT / 'private_contracts.json')['flags']
    assert len(primary) == 254 and len(background) == 69
    assert sum(flags[str(r['qid'])]['C'] for r in primary) == 251
    assert sum(flags[str(r['qid'])]['C'] for r in background) == 21
    assert all(not flags[str(r['qid'])]['C'] for k in ('old_source', 'old_validation', 'old_acquisition') for r in m[k])
    images = sorted({r['img_name'] for r in rows if flags[str(r['qid'])]['C']})
    primary_images = sorted({r['img_name'] for r in primary if flags[str(r['qid'])]['C']})
    assert len(images) >= 20
    shuffled = images.copy()
    random.Random(p['selection_seed']).shuffle(shuffled)
    wrong = {name: shuffled[(i + 1) % len(shuffled)] for i, name in enumerate(shuffled)}
    assert all(a != b for a, b in wrong.items()) and set(wrong.values()) == set(images)
    logical = [{'seed': seed, 'qid': r['qid'], 'condition': condition} for seed in p['expert_seeds'] for r in rows for condition in ('N', 'D', 'B', 'W')]
    random.Random(p['selection_seed']).shuffle(logical)
    for kind, raw in [('modality', 'liver'), ('plane', 'heart'), ('modality', 'CT and liver'), ('plane', 'axial')]:
        assert parse(kind, raw) == ('Transverse' if raw == 'axial' else 'Unknown')
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_manifest.json', {'validation': primary, 'background': background, 'primary_images': primary_images, 'before_flags': {str(r['qid']): flags[str(r['qid'])] for r in rows}, 'admitted_images': images, 'admit': {str(r['qid']): flags[str(r['qid'])]['C'] for r in rows}, 'wrong_image': wrong})
    write(OUT / 'logical_plan.json', logical)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    sample = primary[0]
    wrapped = common.prompt(processor, reader_query(sample, ('Unknown', 'Unknown')))
    original = common.prompt(processor, query_view(sample))
    assert processor.tokenizer(wrapped)['input_ids'] != processor.tokenizer(original)['input_ids']
    assert not torch.cuda.is_initialized()
    expert = len(p['expert_seeds']) * (2 * len(images) + 2)
    upper = expert + 7 * sum(flags[str(r['qid'])]['C'] for r in rows)
    p.update(status='FROZEN_CPU_READY', formal_generation_upper_bound=upper, maximum_generation_calls=upper + 100)
    write(CONFIG, p)
    cpu = {'status': 'CPU_READY', 'validation_questions': 254, 'validation_images': len({r['img_name'] for r in primary}), 'background_questions': 69, 'changed_background_questions': 21, 'admitted_questions': 272, 'admitted_images': len(images), 'uncovered_questions_kept': 3, 'fixed_point_free_wrong_mapping': True, 'expert_generation_calls': expert, 'reader_logical_outcomes': len(logical), 'reader_physical_upper_bound': 7 * 272, 'total_generation_upper_bound': upper, 'generation_cap': upper + 100, 'exact_equal_reader_inputs_deduplicated': True, 'parser_rejects_organ_names': True, 'reader_prompt_differs_from_historical': True, 'gpu_initialized': False}
    write(OUT / 'cpu_check.json', cpu)
    print(json.dumps(cpu), flush=True)


def run():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    import inspect
    inspect.signature(PeftModel.load_adapter).bind_partial(None, Path('.'), adapter_name='probe', is_trainable=False, local_files_only=True)
    assert callable(PeftModel.disable_adapter)
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    torch.set_num_threads(4)
    m = read(OUT / 'private_manifest.json')
    rows = {r['qid']: r for r in m['validation'] + m['background']}
    representatives = {name: next(r for r in rows.values() if r['img_name'] == name) for name in m['admitted_images']}
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = None
    calls = 0

    def generate(query, image, stage, metadata, limit):
        nonlocal calls
        check()
        calls += 1
        assert calls <= p['maximum_generation_calls']
        inputs = processor(text=[common.prompt(processor, query)], images=[image], return_tensors='pt').to('cuda')
        intent = dict(metadata, number=calls, stage=stage, event='intent', started_unix=time.time())
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(intent) + '\n')
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=limit, use_cache=True)
        tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
        record = dict(metadata, number=calls, stage=stage, raw=processor.tokenizer.decode(tokens, skip_special_tokens=True), output_ids=tokens, completed_unix=time.time())
        write(OUT / 'records' / f'{calls:04d}.json', record)
        with (OUT / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(dict(record, event='result')) + '\n')
        write(OUT / 'progress.json', {'stage': stage, 'new_generation_calls': calls, 'upper_bound': p['formal_generation_upper_bound'], 'updated_unix': time.time()})
        return record

    tags, blank = {}, {}
    for index, seed in enumerate(p['expert_seeds']):
        if model is None:
            model = PeftModel.from_pretrained(base, PARENT / f'I_{seed}/adapter', is_trainable=False, local_files_only=True)
            adapter = 'default'
        else:
            adapter = f'seed_{seed}'
            model.load_adapter(PARENT / f'I_{seed}/adapter', adapter_name=adapter, is_trainable=False, local_files_only=True)
        model.set_adapter(adapter)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        for name in [None] + m['admitted_images']:
            row = representatives[name] if name else representatives[m['admitted_images'][0]]
            view = 'N' if name else 'B'
            pair = []
            for kind in QUERIES:
                query = {'qid': -1 if kind == 'modality' else -2, 'img_name': row['img_name'], 'question': QUERIES[kind]}
                result = generate(query, common.image_view(row, view), 'expert', {'expert_seed': seed, 'image_group': m['admitted_images'].index(name) if name else -1, 'kind': kind, 'view': view}, 8)
                pair.append(parse(kind, result['raw']))
            if name:
                tags[seed, name] = tuple(pair)
            else:
                blank[seed] = tuple(pair)
    assert calls == read(OUT / 'cpu_check.json')['expert_generation_calls']
    feasibility = {}
    for seed in p['expert_seeds']:
        joint = [name for name in m['primary_images'] if tags[seed, name] != blank[seed] and tags[seed, name] != tags[seed, m['wrong_image'][name]]]
        distinct_valid = {tags[seed, name] for name in m['primary_images'] if 'Unknown' not in tags[seed, name]}
        feasibility[str(seed)] = {'joint_contrast_images': len(joint), 'distinct_fully_parsed_pairs': len(distinct_valid), 'unknown_field_count': sum(t == 'Unknown' for name in m['admitted_images'] for t in tags[seed, name]), 'pass': len(joint) >= 20 and len(distinct_valid) >= 3}
    write(OUT / 'EXPERT_COMPLETED.json', {'expert_generation_calls': calls, 'feasibility': feasibility})
    if not all(r['pass'] for r in feasibility.values()):
        write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_DYNAMIC_ACQUISITION_TOOLS', 'feasibility_pass': False, 'new_generation_calls': calls, 'expert_generations': calls, 'reader_generations': 0, 'logical_reader_outcomes': 0, 'optimizer_steps': 0, 'test_annotation_opened': False, 'completed_unix': time.time()})
        return
    expert_calls = calls
    logical = read(OUT / 'logical_plan.json')
    physical, cached, resolved = [], {}, []
    for i, e in enumerate(logical):
        row, seed, condition = rows[e['qid']], e['seed'], e['condition']
        pair = ('Unknown', 'Unknown')
        if m['admit'][str(e['qid'])]:
            if condition == 'D': pair = tags[seed, row['img_name']]
            elif condition == 'B': pair = blank[seed]
            elif condition == 'W': pair = tags[seed, m['wrong_image'][row['img_name']]]
        if not m['admit'][str(e['qid'])]:
            flag = m['before_flags'][str(e['qid'])]
            prior = f'A_{seed}' if flag['A'] else (f'I_{seed}' if flag['B'] else 'base')
            resolved.append(dict(e, logical_index=i, reference_model=prior))
            continue
        key = (row['img_name'], row['question'], pair)
        if key not in cached:
            cached[key] = len(physical)
            physical.append({'qid': row['qid'], 'tags': pair})
        resolved.append(dict(e, logical_index=i, physical_index=cached[key]))
    assert len(physical) <= 7 * sum(m['admit'].values())
    write(OUT / 'private_physical_reader_plan.json', physical)
    write(OUT / 'private_resolved_logical_plan.json', resolved)
    with model.disable_adapter():
        layers = [x for x in model.modules() if hasattr(x, 'disable_adapters')]
        assert layers and all(x.disable_adapters for x in layers)
        write(OUT / 'BASE_READER_CHECK.json', {'all_lora_layers_disabled': True, 'layers': len(layers), 'reader_optimizer_steps': 0})
        for i, e in enumerate(physical):
            row = rows[e['qid']]
            generate(reader_query(row, e['tags']), common.image_view(row, 'N'), 'reader', {'physical_index': i, 'qid': e['qid']}, 64)
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_DYNAMIC_ACQUISITION_TOOLS', 'feasibility_pass': True, 'new_generation_calls': calls, 'expert_generations': expert_calls, 'reader_generations': calls - expert_calls, 'logical_reader_outcomes': len(logical), 'optimizer_steps': 0, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
