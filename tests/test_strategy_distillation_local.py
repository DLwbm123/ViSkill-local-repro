
def test_local_policy_strategy_is_strictly_parseable(strategy):
    assert strategy['fixture_only'] and strategy['strict_tags']
    assert len(strategy['parsed_strategy'].split())>=8
    assert 'when' in strategy['parsed_strategy'].lower()
    assert any(a in strategy['parsed_strategy'].lower() for a in ['right','left','down','up','move','navigate'])
