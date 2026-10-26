"""Training-side feasibility of the proposed image-isolated, modality-supported target task."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
ROOT = Path(os.environ['TASK_ROOT'])
SRC = Path(os.environ['PAPER_SOURCE'])
DATA = Path(os.environ['SLAKE_ROOT'])
sys.path.insert(0, str(SRC / 'scripts'))
from medical_region_reader import read, write
from medical_roi_task_audit import task, match_target
from slake_probe import normalize


def audit():
    m = read(ROOT / 'artifacts/medical/continual_acquisition_bank_v1/private_manifest.json')
    c = read(ROOT / 'artifacts/medical/continual_acquisition_bank_v1/private_contracts.json')
    history = {r['img_name'] for k in ['train', 'old_train_replay'] for r in m[k]}
    reserved = {r['img_name'] for k in ['old_source', 'old_background'] for r in m[k]}
    pairs, lexicon = {tuple(x) for x in c['A_pairs']}, c['A_lexicon']
    b = set(c['B_normalized_question_templates'])
    train = [r for r in read(DATA / 'raw/train.json') if r['q_lang'] == 'en' and r['base_type'] == 'vqa' and r['content_type'] == 'Organ' and (task(r['question']), match_target(r['question'], lexicon)) not in pairs and normalize(r['question']) not in b]
    pools = {'outside_reserved_source_evaluation': [r for r in train if r['img_name'] not in reserved], 'source_target_image_isolated': [r for r in train if r['img_name'] not in reserved | history]}
    stats = {name: {'questions': len(pool), 'unique_images': len({r['img_name'] for r in pool}), 'questions_by_modality': dict(Counter(r['modality'] for r in pool)), 'unique_images_by_modality': {kind: len({r['img_name'] for r in pool if r['modality'] == kind}) for kind in ['CT', 'MRI', 'X-Ray']}, 'question_templates': len({normalize(r['question']) for r in pool})} for name, pool in pools.items()}
    candidate = stats['source_target_image_isolated']
    enough = candidate['questions'] >= 100 and all(candidate['unique_images_by_modality'][kind] >= 10 for kind in ['CT', 'MRI', 'X-Ray'])
    result = {'status': 'CPU_DATA_FEASIBILITY_PASS' if enough else 'CPU_DATA_FEASIBILITY_STOP', 'source_history_unique_images': len(history), 'reserved_source_evaluation_images': len(reserved), 'pools': stats, 'required_minimum_target_images_each_modality': 10, 'planned_target_questions': 100, 'image_or_question_correctness_used_for_selection': False, 'model_calls': 0, 'optimizer_steps': 0, 'test_gold_read': False, 'new_training_allowed_by_previous_readout_gate': False, 'limits': ['Only training-side input/data feasibility; no new candidate or test evaluation.', 'The10image-per-modality requirement comes from the reviewed plan, not a clinical power calculation.', 'If the source-isolated X-ray pool is insufficient, repeated questions on the same image cannot fix it.', 'A data-feasibility pass would not override the previous readout-gate failure or authorize F52/F104/R52.']}
    output = ROOT / 'artifacts/medical/isolated_target_feasibility_v1'
    output.mkdir(exist_ok=False)
    write(output / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': audit()
