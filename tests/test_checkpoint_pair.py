"""CPU persistence tests; dummy model files do not establish real PPO restore."""
import json
from pathlib import Path
import pytest
from PIL import Image
from vagen.skills.checkpoint_pair import save_library_pair,validate_library_pair,restore_library_pair
from vagen.skills.features import FrozenLakeFeatureExtractor
from vagen.skills.library import ViSkillLibrary


def setup_pair(tmp_path):
    active=tmp_path/'active';ckpt=tmp_path/'global_step_2'
    for role in ['actor','critic']:
        (ckpt/role).mkdir(parents=True)
        (ckpt/role/'TEST_ONLY_dummy_model').write_text(role)
    lib=ViSkillLibrary(FrozenLakeFeatureExtractor())
    lib.admit({'features':[0.]*10,'strategy':'test only','trajectory':['right'],'positions':[[0,0],[0,1]],'image':Image.new('RGB',(64,64),'blue')})
    lib.update_utility('skill_0',1.);lib.save(active)
    return active,ckpt,lib


def test_snapshot_is_immutable_and_restores_metadata_with_backup(tmp_path):
    active,ckpt,lib=setup_pair(tmp_path)
    save_library_pair(ckpt,active,['actor','critic'])
    lib.update_utility('skill_0',0.);lib.save(active)
    receipt=restore_library_pair(ckpt,active,['actor','critic'])
    lib.load(active)
    assert lib._skills[0]['utility']==pytest.approx(.55)
    assert lib._skills[0]['use_count']==1
    assert lib._skills[0]['positions']==[[0,0],[0,1]]
    assert lib.load_image(lib._skills[0],active).getpixel((0,0))==(0,0,255)
    backup=json.loads((Path(receipt['previous_library_backup'])/'skill_index.json').read_text())
    assert backup['skills'][0]['utility']==pytest.approx(.495)
    assert backup['skills'][0]['use_count']==2


def test_mismatched_step_fails_without_mutating_active_library(tmp_path):
    active,ckpt,lib=setup_pair(tmp_path);save_library_pair(ckpt,active,['actor','critic'])
    marker=ckpt/'skill_pair.json';data=json.loads(marker.read_text());data['global_step']=3;marker.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='mismatch'):restore_library_pair(ckpt,active,['actor','critic'])
    lib.load(active);assert lib._skills[0]['use_count']==1
    assert not list(tmp_path.glob('active.before_restore_*'))


def test_incomplete_pair_fails_closed(tmp_path):
    active,ckpt,lib=setup_pair(tmp_path)
    with pytest.raises(FileNotFoundError):validate_library_pair(ckpt,['actor','critic'])
    save_library_pair(ckpt,active,['actor','critic'])
    (ckpt/'paired_skill_library/skill_0.png').unlink()
    with pytest.raises(ValueError,match='image'):validate_library_pair(ckpt,['actor','critic'])


def test_save_rejects_missing_model_before_publishing_pair(tmp_path):
    active,ckpt,lib=setup_pair(tmp_path)
    (ckpt/'critic/TEST_ONLY_dummy_model').unlink()
    with pytest.raises(ValueError,match='model role'):save_library_pair(ckpt,active,['actor','critic'])
    assert not (ckpt/'skill_pair.json').exists()


def test_empty_no_cold_start_library_can_pair(tmp_path):
    ckpt=tmp_path/'global_step_1';(ckpt/'actor').mkdir(parents=True);(ckpt/'actor/dummy').touch()
    active=tmp_path/'initially_absent'
    assert save_library_pair(ckpt,active,['actor'])['skill_count']==0
    assert restore_library_pair(ckpt,active,['actor'])['skill_count']==0
    assert json.loads((active/'skill_index.json').read_text())['skills']==[]


def test_restore_rejects_ancestor_active_path(tmp_path):
    active,ckpt,lib=setup_pair(tmp_path);save_library_pair(ckpt,active,['actor','critic'])
    with pytest.raises(ValueError,match='active library'):
        restore_library_pair(ckpt,tmp_path,['actor','critic'])
    assert (ckpt/'skill_pair.json').exists()
