"""Experiment-only reference-content controls; environment and retrieval stay fixed."""
from PIL import Image
from .bootstrap_agent import BootstrapGymAgentLoop
from vagen.skills.provider import SKILL_INTRO


def reference_observation(condition, original, text):
    if condition == 'V':
        return original
    if condition == 'Blank':
        images = original['multi_modal_input']['<image>']
        assert len(images) == 1
        blank = Image.new('RGB', images[0].size, 'white')
        return {'obs_str': original['obs_str'], 'multi_modal_input': {'<image>': [blank]}}
    if condition == 'Text':
        assert '<image>' not in text and text
        return {'obs_str': SKILL_INTRO + text}
    raise ValueError(condition)


class CardControlAgent(BootstrapGymAgentLoop):
    def _inject_skill(self, agent_data, skill, **kwargs):
        original = self.skill_provider.format_for_prompt
        observed = original(skill)
        controlled = reference_observation(self.card_condition, observed, self.card_reference_text)
        self.card_reference_audit = {'condition': self.card_condition,
            'obs_str': controlled['obs_str'],
            'image_sizes': [list(x.size) for x in controlled.get('multi_modal_input', {}).get('<image>', [])]}
        self.skill_provider.format_for_prompt = lambda _: controlled
        try:
            return super()._inject_skill(agent_data, skill, **kwargs)
        finally:
            self.skill_provider.format_for_prompt = original
