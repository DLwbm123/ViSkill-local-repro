from conftest import ART
from vagen.skills.library import ViSkillLibrary


def test_full_library_roundtrip(provider,skill_data):
    assert len(provider.library)==0
    sid=provider.library.admit(skill_data)
    path=ART/'roundtrip_library';provider.library.save(path)
    loaded=ViSkillLibrary(provider.library.feature_extractor);loaded.load(path)
    assert len(loaded)==1
    skill=loaded._skills[0]
    assert skill['id']==sid and skill['trajectory']==skill_data['trajectory']
    assert skill['features']==skill_data['features'] and skill['strategy']==skill_data['strategy']
    assert skill['utility']==.5 and skill['use_count']==0
    image=loaded.load_image(skill,path);assert image and image.size==skill_data['image'].size
