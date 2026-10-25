"""Train-only intent admission and prospective replay of deterministic adapter routing."""
from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
NAME = 'task_supported_adapter_v1'
OUT = ROOT / 'artifacts/medical' / NAME
CONFIG = SRC / 'configs/reproduction' / (NAME + '.json')
PREVIOUS = ROOT / 'artifacts/medical/grounded_reader_anchor_v1'
sys.path.insert(0, str(SRC / 'scripts'))
import medical_region_reader as common
from medical_roi_task_audit import task, match_target
from slake_probe import score, query_view
read, write = common.read, common.write


def check():
    assert time.time() < read(CONFIG)['cooperative_stop_unix']
    assert not (OUT / 'STOP_REQUESTED.json').exists()
    assert not (ROOT / 'state/medical_window_12h_20261025/STOP_NEW_WORK.json').exists()


def terms(question):
    assert isinstance(question, str)
    words = re.findall(r'[a-z0-9]+', question.lower())
    return Counter(['w:' + w for w in words] + ['b:' + a + ' ' + b for a, b in zip(words, words[1:])])


def vectorize(questions, vocabulary, idf):
    index = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(questions), len(vocabulary)), dtype=np.float64)
    for i, question in enumerate(questions):
        for word, count in terms(question).items():
            if word in index:
                matrix[i, index[word]] = np.log1p(count) * idf[index[word]]
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def prepare():
    import torch
    from transformers import AutoProcessor
    p = read(CONFIG)
    assert p['status'] == 'DRAFT_FOR_CPU' and not OUT.exists()
    check()
    old = read(PREVIOUS / 'private_manifest.json')
    assert read(PREVIOUS / 'RESULT.json')['status'] == 'COMPLETE_VISUAL_ANCHOR_NOT_ESTABLISHED'
    exposed = {r['img_name'] for r in old['train'] + old['replay']}
    pairs = {(r['task'], r['target']) for r in old['train']}
    lexicon = sorted({target for kind, target in pairs}, key=lambda x: (-len(x), x))
    def eligible(row):
        return (task(row['question']), match_target(row['question'], lexicon)) in pairs
    rows = [r for r in read(DATA / 'raw/train.json') if r['q_lang'] == 'en' and r['base_type'] == 'vqa']
    fit = [{**{k: r[k] for k in ('qid', 'img_name', 'question')}, 'admissible': eligible(r)} for r in rows if r['img_name'] in exposed]
    remaining = [r for r in rows if r['img_name'] not in exposed]
    positive, negative = defaultdict(list), defaultdict(list)
    for row in remaining:
        (positive if eligible(row) else negative)[row['img_name']].append(row)
    rng = random.Random(p['selection_seed'])
    primary = [rng.choice(sorted(pool, key=lambda r: r['qid'])) for name, pool in sorted(positive.items())]
    for row in primary:
        row['task'] = task(row['question'])
        row['target'] = match_target(row['question'], lexicon)
    names = sorted(set(negative) - set(positive))
    rng.shuffle(names)
    retention = [rng.choice(sorted(negative[name], key=lambda r: r['qid'])) for name in names[:len(primary)]]
    assert len(fit) == 1569 and sum(r['admissible'] for r in fit) == 124
    assert len(primary) == len(retention) == 69
    assert len({r['img_name'] for r in primary + retention}) == 138
    assert not exposed & {r['img_name'] for r in primary + retention}
    OUT.mkdir(parents=True)
    (OUT / 'records').mkdir()
    write(OUT / 'private_manifest.json', {'fit': fit, 'primary': primary, 'retention': retention})
    write(OUT / 'private_support.json', {'task_target_pairs': sorted(pairs), 'lexicon': lexicon})
    schedule = [{'panel': kind, 'qid': row['qid'], 'view': view} for kind, pool in (('primary', primary), ('retention', retention)) for row in pool for view in ('N', 'B')]
    rng.shuffle(schedule)
    assert len(schedule) == 276
    write(OUT / 'schedule.json', schedule)
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    row = primary[0]
    text = common.prompt(processor, query_view(row))
    images = [common.image_view(row, view) for view in ('N', 'B')]
    encoded = [processor(text=[text], images=[image], return_tensors='pt') for image in images]
    assert torch.equal(encoded[0]['input_ids'], encoded[1]['input_ids'])
    assert torch.equal(encoded[0]['image_grid_thw'], encoded[1]['image_grid_thw'])
    fixture = vectorize(['a a', 'b', 'unknown'], ['w:a', 'w:b'], np.ones(2))
    assert np.allclose(fixture, [[1, 0], [0, 1], [0, 0]])
    assert 'answer' not in fit[0] and not torch.cuda.is_initialized()
    write(OUT / 'cpu_check.json', {'status': 'CPU_READY', 'router_fit_questions': len(fit), 'router_fit_images': len(exposed), 'router_fit_positive': 124, 'primary_questions': 69, 'primary_images': 69, 'primary_tasks': dict(Counter(r['task'] for r in primary)), 'retention_questions': 69, 'retention_images': 69, 'all_evaluation_images_excluded_from_adapter_and_router_updates': True, 'fit_contains_no_answer_fields': True, 'matched_N_Blank_ids_grids': True, 'zero_vector_and_normalization_fixture': True, 'gpu_initialized': False, 'new_generation_calls': 828})
    p['status'] = 'FROZEN_CPU_READY'
    write(CONFIG, p)
    print(json.dumps(read(OUT / 'cpu_check.json')), flush=True)


def run():
    import torch
    import torch.nn.functional as F
    p = read(CONFIG)
    assert p['status'] == 'FROZEN_CPU_READY' and not (OUT / 'requests.jsonl').exists()
    check()
    torch.set_num_threads(4)
    m = read(OUT / 'private_manifest.json')
    fit = m['fit']
    df = Counter(word for row in fit for word in terms(row['question']))
    vocabulary = sorted((word for word, count in df.items() if count >= 2), key=lambda word: (-df[word], word))[:4096]
    idf = np.array([np.log((len(fit) + 1) / (df[word] + 1)) + 1 for word in vocabulary])
    features = torch.from_numpy(vectorize([r['question'] for r in fit], vocabulary, idf))
    labels = torch.tensor([r['admissible'] for r in fit], dtype=torch.float64)
    coefficient = torch.nn.Parameter(torch.zeros(len(vocabulary) + 1, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([coefficient], lr=1, max_iter=100, max_eval=150, tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn='strong_wolfe')
    positive_weight = (len(fit) - labels.sum()) / labels.sum()
    evaluations = []
    def closure():
        check()
        optimizer.zero_grad(set_to_none=True)
        logits = features @ coefficient[:-1] + coefficient[-1]
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight) + coefficient[:-1].square().sum() / (2 * len(fit))
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(coefficient.grad).all()
        evaluations.append(float(loss.detach()))
        return loss
    optimizer.step(closure)
    assert float(coefficient.detach().square().sum()) > 0
    state = optimizer.state[coefficient]
    torch.save({'coefficient': coefficient.detach(), 'idf': torch.from_numpy(idf), 'vocabulary': vocabulary}, OUT / 'private_router.pt')
    write(OUT / 'ROUTER_TRAIN_COMPLETED.json', {'fit_questions': len(fit), 'fit_images': len({r['img_name'] for r in fit}), 'fit_positives': int(labels.sum()), 'features': len(vocabulary), 'parameters': coefficient.numel(), 'LBFGS_iterations': state['n_iter'], 'closure_evaluations': len(evaluations), 'initial_objective': evaluations[0], 'final_objective': evaluations[-1], 'finite_gradient': True, 'coefficient_square_norm': float(coefficient.detach().square().sum()), 'evaluation_answers_used': False, 'completed_unix': time.time()})
    evaluation = sorted(m['primary'] + m['retention'], key=lambda r: r['qid'])
    with torch.no_grad():
        vectors = torch.from_numpy(vectorize([r['question'] for r in evaluation], vocabulary, idf))
        probabilities = torch.sigmoid(vectors @ coefficient[:-1] + coefficient[-1]).tolist()
    admitted = [prob >= p['admission_threshold'] for prob in probabilities]
    sham = admitted.copy()
    random.Random(p['selection_seed'] + 1).shuffle(sham)
    assert sum(sham) == sum(admitted)
    support = read(OUT / 'private_support.json')
    supported_pairs = {tuple(pair) for pair in support['task_target_pairs']}
    primary_ids = {r['qid'] for r in m['primary']}
    routing = {}
    for row, prob, flag, control in zip(evaluation, probabilities, admitted, sham):
        rule = (task(row['question']), match_target(row['question'], support['lexicon'])) in supported_pairs
        assert rule == (row['qid'] in primary_ids)
        routing[str(row['qid'])] = {'probability': prob, 'admit': flag, 'sham_admit': control, 'rule_admit': rule}
    write(OUT / 'private_routing_frozen.json', {'threshold': p['admission_threshold'], 'queries': routing, 'frozen_before_new_model_generations': True, 'admitted': sum(admitted), 'sham_admitted': sum(sham)})
    del optimizer, features, labels, vectors, coefficient
    gc.collect()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(ROOT / 'models/b0', local_files_only=True, max_pixels=401408)
    processor.tokenizer.padding_side = 'right'
    items = {r['qid']: r for r in evaluation}
    plan = read(OUT / 'schedule.json')
    calls = 0
    for label in ('base', 'A_42', 'A_43'):
        check()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT / 'models/b0', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        if label != 'base':
            model = PeftModel.from_pretrained(model, PREVIOUS / label / 'adapter', is_trainable=False, local_files_only=True)
        model.eval()
        for entry in plan:
            check()
            calls += 1
            assert calls <= p['maximum_generation_calls']
            row = items[entry['qid']]
            text = common.prompt(processor, query_view(row))
            inputs = processor(text=[text], images=[common.image_view(row, entry['view'])], return_tensors='pt').to('cuda')
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(entry, event='intent', number=calls, model=label, started_unix=time.time())) + '\n')
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
            tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
            raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            record = dict(entry, number=calls, model=label, raw=raw, output_ids=tokens, correct=bool(score(raw, row['answer'])), completed_unix=time.time())
            write(OUT / 'records' / f'{calls:04d}.json', record)
            with (OUT / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(record, event='result')) + '\n')
            write(OUT / 'progress.json', {'phase': 'new_question_evaluation', 'model': label, 'new_generation_calls': calls, 'planned_calls': 828, 'updated_unix': time.time()})
        write(OUT / f'{label}_EVALUATED.json', {'formal_calls': 276, 'completed_unix': time.time()})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert calls == 828
    write(OUT / 'WORKER_COMPLETED.json', {'status': 'COMPLETE_TASK_SUPPORT_ACTIVATION', 'new_generation_calls': calls, 'new_reader_optimizer_steps': 0, 'new_router_fitted': True, 'counterfactual_policy_responses_share_exact_branch_records': True, 'test_annotation_opened': False, 'completed_unix': time.time()})


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[os.environ['TASK_STAGE']]()
