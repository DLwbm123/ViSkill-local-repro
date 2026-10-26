"""Correct global versus consumer-specific image exposure, including the replay key."""
import json
import os
from pathlib import Path
ROOT = Path(os.environ['TASK_ROOT'])
BASE = ROOT / 'artifacts/medical'
read = lambda name: json.loads((BASE / name / 'private_manifest.json').read_text())


def audit():
    a, i, c, g = [read(n) for n in ['grounded_reader_anchor_v1', 'continual_acquisition_bank_v1', 'acquisition_to_organ_transfer_v1', 'visual_consensus_guard_v1']]
    images = lambda rows: {r['img_name'] for r in rows}
    primary, target = images(g['primary']), images(c['train'])
    a_history = images(a['train'] + a['replay'])
    i_history = images(i['train'] + i['old_train_replay'])
    ic_history = i_history | target
    all_training = set()
    for name in ['region_reader_learning_v1', 'grounded_reader_anchor_v1', 'continual_acquisition_bank_v1', 'acquisition_to_organ_transfer_v1']:
        m = read(name)
        for key in ['train', 'replay', 'old_train_replay']: all_training |= images(m.get(key, []))
    raw = Path(os.environ['SLAKE_ROOT']) / 'raw'
    test = json.loads((raw / 'test.json').read_text(), object_pairs_hook=lambda pairs: {k: v for k, v in pairs if k != 'answer'})
    test_images = images([r for r in test if r['q_lang'] == 'en'])
    result = {'status': 'COMPLETE_CONSUMER_AWARE_EXPOSURE_CORRECTION', 'previous_omitted_manifest_key': 'grounded_reader_anchor_v1.replay', 'primary85_images': len(primary), 'A_parent_training_images': len(a_history), 'A_KL_replay_images': len(images(a['replay'])), 'I_parent_training_images': len(i_history), 'I_plus_C_training_images': len(ic_history), 'all_window_training_images_including_replay': len(all_training), 'primary85_overlap_A_KL_replay': len(primary & images(a['replay'])), 'primary85_overlap_all_window_training': len(primary & all_training), 'primary85_overlap_I_parent_training': len(primary & i_history), 'primary85_overlap_I_plus_C_training': len(primary & ic_history), 'C_training_overlap_A_KL_replay': len(target & images(a['replay'])), 'C_training_overlap_I_parent_training': len(target & i_history), 'locked_test_images': len(test_images), 'locked_test_overlap_A_parent_training': len(test_images & a_history), 'locked_test_overlap_I_parent_training': len(test_images & i_history), 'locked_test_overlap_all_window_training': len(test_images & all_training), 'model_calls': 0, 'optimizer_steps': 0, 'test_gold_read': False, 'score_or_gate_changes': False, 'limits': ['The85panel is isolated from its actual I/C parent updates, not all unrelated model training in the window.', 'The locked96test images remain isolated by image ID from all these parent histories; original near-duplicate and patient limitations remain.', 'This audit supersedes the global0intersection field in history_exposure_audit_v1; source-specific C/I overlap40 is unchanged.']}
    assert result['primary85_overlap_all_window_training'] == 32 and result['primary85_overlap_I_plus_C_training'] == 0
    assert result['locked_test_overlap_all_window_training'] == 0
    out = BASE / 'consumer_exposure_audit_v2'
    out.mkdir(exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': audit()
