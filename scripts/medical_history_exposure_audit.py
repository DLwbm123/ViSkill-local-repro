"""Anonymous image-ID exposure audit including nonzero old-task KL inputs."""
import json
import os
from pathlib import Path
ROOT = Path(os.environ['TASK_ROOT'])
BASE = ROOT / 'artifacts/medical'
read = lambda p: json.loads(p.read_text())


def audit():
    c = read(BASE / 'acquisition_to_organ_transfer_v1/private_manifest.json')
    i = read(BASE / 'continual_acquisition_bank_v1/private_manifest.json')
    g = read(BASE / 'visual_consensus_guard_v1/private_manifest.json')
    for seed in [42, 43]:
        assert read(BASE / f'continual_acquisition_bank_v1/I_{seed}/TRAIN_COMPLETED.json')['old_coefficient'] == 1
    items = {r['img_name']: r['modality'] for m in [c, i, g] for rows in m.values() if isinstance(rows, list) for r in rows if isinstance(r, dict) and 'img_name' in r}
    ct, supervised, old_kl = [set(r['img_name'] for r in rows) for rows in [c['train'], i['train'], i['old_train_replay']]]
    sets = {'C_target_train': ct, 'I_acquisition_supervised': supervised, 'I_nonzero_old_KL': old_kl, 'I_all_backward_images': supervised | old_kl, 'C_intersection_I_supervised': ct & supervised, 'C_intersection_I_old_KL': ct & old_kl, 'C_intersection_I_all_backward': ct & (supervised | old_kl), 'C_intersection_both_I_sources': ct & supervised & old_kl}
    stats = {k: {'unique_images': len(v), 'modality_unique_images': {m: sum(items[n] == m for n in v) for m in sorted({items[n] for n in v})}} for k, v in sets.items()}
    full_history = set()
    for name in ['region_reader_learning_v1', 'grounded_reader_anchor_v1', 'continual_acquisition_bank_v1', 'acquisition_to_organ_transfer_v1']:
        manifest = read(BASE / name / 'private_manifest.json')
        for kind in ['train', 'old_train_replay']: full_history.update(r['img_name'] for r in manifest.get(kind, []))
    primary = {r['img_name'] for r in g['primary']}
    assert len(primary) == 85 and not primary & full_history
    assert stats['C_intersection_I_supervised']['unique_images'] == 22 and stats['C_intersection_I_old_KL']['unique_images'] == 26 and stats['C_intersection_I_all_backward']['unique_images'] == 40
    result = {'status': 'COMPLETE_ANONYMOUS_IMAGE_ID_EXPOSURE_AUDIT', 'counts': stats, 'both_I_seed_old_KL_coefficient': 1, 'frozen_readout_primary_images': 85, 'primary_intersection_all_window_training_images': 0, 'checks': {'C_I_union_overlap_equals_40_not_22': True, 'primary85_disjoint_from_supervised_and_old_KL': True}, 'limits': ['Image IDs only; pixel/near duplicates and patient identities not assessed.', 'The22count remains correct for acquisition-supervised images only; it is not the full history overlap.', 'This correction does not change frozen outcomes, scores, gates or original training.', 'No test annotations, model operations or training in this audit.']}
    output = BASE / 'history_exposure_audit_v1'
    output.mkdir(exist_ok=False)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': audit()
