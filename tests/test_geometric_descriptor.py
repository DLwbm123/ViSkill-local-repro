import pytest
from vagen.skills.features import FrozenLakeFeatureExtractor


def test_geometry_matches_hand_computation(trajectory):
    f=FrozenLakeFeatureExtractor();x=f.extract(trajectory['state'])
    assert x==[1,1,3,3,0,0,1,0,0,0]
    assert f.similarity(x,x)==1
    y=x.copy();y[0]=-1
    assert f.similarity(x,y)==pytest.approx(.875)
