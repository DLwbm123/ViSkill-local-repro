import pytest
from conftest import save


def test_success_failure_ema_and_use_count(provider,skill_data):
    sid=provider.library.admit(skill_data)
    provider.update_utility(sid,1.);first=provider.library._skills[0]['utility']
    provider.update_utility(sid,0.);second=provider.library._skills[0]['utility']
    save('utility.json',{'initial':.5,'after_success':first,'after_failure':second,'use_count':provider.library._skills[0]['use_count']})
    assert first==pytest.approx(.55) and second==pytest.approx(.495)
    assert provider.library._skills[0]['use_count']==2
