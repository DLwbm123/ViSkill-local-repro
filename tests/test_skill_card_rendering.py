import numpy as np
from conftest import ART
from PIL import Image


def test_card_has_ordered_frames_path_and_strategy(trajectory,strategy):
    assert len(trajectory['frames'])==7 and len(trajectory['actions'])==6
    for a,p,q in zip(trajectory['actions'],trajectory['positions'],trajectory['positions'][1:]):
        expected={'down':(1,0),'right':(0,1),'up':(-1,0),'left':(0,-1)}[a]
        assert (q[0]-p[0],q[1]-p[1])==expected
    im=Image.open(ART/'final_skill_card.png');pixels=np.asarray(im)
    assert im.width>7*120 and im.height>140 and pixels.std()>10
    assert np.count_nonzero((pixels[:20]<80).all(axis=2))>20
    marked=np.asarray(Image.open(ART/'annotated_6.png'))
    original=np.asarray(trajectory['frames'][6])
    assert np.any(marked!=original)
    assert strategy['card_visual_tokens']>0
