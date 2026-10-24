"""Scoped immutable-library context crossing, with per-instance provider choice."""
import json,os
from pathlib import Path
from omegaconf import OmegaConf
from .bootstrap_agent import BootstrapGymAgentLoop
from vagen.skills.provider import ViSkillProvider

class FixedLibraryAgent(BootstrapGymAgentLoop):
    @classmethod
    def init_class(cls,config,tokenizer,processor,**kwargs):
        super().init_class(config,tokenizer,processor,**kwargs)
        if '_fixed_providers' not in cls.__dict__:
            cls._fixed_providers={}
            for context,variable in [('empty','FIXED_EMPTY_LIBRARY'),('cards','FIXED_CARD_LIBRARY')]:
                cfg=OmegaConf.to_container(config.skill_system,resolve=True)
                cfg['library']['path']=os.environ[variable]
                cls._fixed_providers[context]=ViSkillProvider(cfg)

    async def run(self,sampling_params,**kwargs):
        context=kwargs['fixed_context'];assert context in {'empty','cards'}
        self.pilot_context=context
        self.skill_provider=self.__class__._fixed_providers[context]
        output=await super().run(sampling_params,**kwargs)
        row=output.extra_fields
        manifest=json.loads(Path(os.environ['PILOT_DATA_MANIFEST']).read_text())
        target=next(x for x in manifest['splits'][kwargs['pilot_phase']] if x['seed']==int(kwargs['seed']))
        if context=='empty':assert row['retrieved_skill_id'] is None
        else:
            assert len(self.skill_provider.library._skills)==5
            if target.get('category')=='multi_up3_applicable' or target['group']=='conditional':
                assert row['retrieved_skill_id']=='skill_4','predeclared three-step exposure absent'
        row['fixed_context']=context;row['fixed_group']=target['group']
        path=Path(row['pilot_episode_path']);serial=json.loads(path.read_text())
        serial.update(fixed_context=context,fixed_group=target['group'],category=target.get('category'))
        path.write_text(json.dumps(serial))
        return output
