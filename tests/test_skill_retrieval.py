from conftest import save


def test_similar_and_unrelated_threshold(provider,skill_data,trajectory):
    provider.library.admit(skill_data)
    same=provider.library.retrieve(trajectory['state'])
    far={'desc':[list(s) for s in ['GHFF','HFFF','FFHS','FFFF']]}
    other=provider.library.retrieve(far)
    save('retrieval.json',{'same_score':same['score'],'far_score':other['score'],'threshold':provider.min_score})
    assert same['score']>=provider.min_score and other['score']<provider.min_score
