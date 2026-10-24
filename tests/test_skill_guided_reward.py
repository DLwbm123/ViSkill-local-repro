import pytest
from conftest import save
from vagen.skills.reward import compute_reward,compute_prefix_reward


def test_outcome_format_unrelated_and_prefix_breakdown():
    ref=['right','right','down','down']
    assert compute_reward([.02,1.02],'skill_guided',True,ref,ref,.8)==pytest.approx(1.04)
    assert compute_reward([.02],'skill_guided',False,ref,['left'],.8)==pytest.approx(.02)
    assert compute_reward([.02],'skill_guided',False,ref,['right','right'],.8)==pytest.approx(.42)


def test_paper_relative_state_progress_counterexample():
    ref=['right','right','down','down'];model=['down','right','right']
    movements={'right':(0,1),'down':(1,0)}
    def positions(actions):
        out=[(0,0)]
        for a in actions:
            dr,dc=movements[a];out.append((out[-1][0]+dr,out[-1][1]+dc))
        return out
    visited=set(positions(model))
    progress=max(i for i,p in enumerate(positions(ref)) if p in visited)/len(ref)
    actual=compute_prefix_reward(ref,model)
    save('reward_counterexample.json',{'reference':ref,'model':model,'paper_state_progress':progress,
         'official_prefix':actual,'paper_total':.02+.8*progress,'official_total':.02+.8*actual})
    assert actual==pytest.approx(progress), 'published prefix reward differs from paper state-progress'
