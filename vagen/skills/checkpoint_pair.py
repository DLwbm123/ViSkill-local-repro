"""Opt-in local checkpoint/skill pairing. No model or optimizer serialization here."""
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4


def _step(folder):
    name = Path(folder).name
    if not name.startswith('global_step_'):
        raise ValueError('pair requires a global_step_N checkpoint directory')
    return int(name.removeprefix('global_step_'))


def _index(folder):
    folder = Path(folder)
    data = json.loads((folder / 'skill_index.json').read_text())
    skills = data['skills']
    ids = [skill['id'] for skill in skills]
    if len(ids) != len(set(ids)) or not isinstance(data['next_id'], int):
        raise ValueError('invalid skill index')
    for skill in skills:
        image = folder / skill['image_path']
        if (not skill['image_path'] or not image.resolve().is_relative_to(folder.resolve())
                or not image.is_file() or image.stat().st_size == 0):
            raise ValueError('missing or unsafe visual skill image')
    return data


def _copy_library(source, target):
    source, target = Path(source), Path(target)
    data = _index(source)
    target.mkdir()
    shutil.copy2(source / 'skill_index.json', target / 'skill_index.json')
    for skill in data['skills']:
        relative = skill['image_path']
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target / relative)
    return data


def save_library_pair(checkpoint, library, model_roles):
    """Publish pair marker only after model saves and a complete immutable snapshot."""
    checkpoint, library = Path(checkpoint), Path(library)
    step = _step(checkpoint)
    for role in model_roles:
        if not (checkpoint / role).is_dir() or not any((checkpoint / role).iterdir()):
            raise ValueError(f'model role missing: {role}')
    target = checkpoint / 'paired_skill_library'
    if target.exists() or (checkpoint / 'skill_pair.json').exists():
        raise FileExistsError('refuse to replace an existing checkpoint/library pair')
    stage = checkpoint / ('.skill_stage_' + uuid4().hex)
    if (library / 'skill_index.json').exists():
        data = _copy_library(library, stage)
    else:
        if library.exists() and any(library.iterdir()):
            raise ValueError('active library is nonempty but has no index')
        stage.mkdir()
        data = {'next_id': 0, 'skills': []}
        (stage / 'skill_index.json').write_text(json.dumps(data))
    _index(stage)
    os.replace(stage, target)
    marker = {'schema': 1, 'global_step': step, 'library': target.name,
              'skill_count': len(data['skills']), 'model_roles': list(model_roles),
              'scope': 'weights and skill snapshot pairing; optimizer continuity depends on save_contents'}
    tmp = checkpoint / '.skill_pair.tmp'
    tmp.write_text(json.dumps(marker, indent=2))
    os.replace(tmp, checkpoint / 'skill_pair.json')
    return marker


def validate_library_pair(checkpoint, model_roles):
    """Fail before model loading if sidecar is absent, mismatched, or incomplete."""
    checkpoint = Path(checkpoint)
    marker = json.loads((checkpoint / 'skill_pair.json').read_text())
    if (marker['schema'] != 1 or marker['global_step'] != _step(checkpoint)
            or marker['library'] != 'paired_skill_library'
            or marker['model_roles'] != list(model_roles)):
        raise ValueError('checkpoint/library pair metadata mismatch')
    for role in model_roles:
        if not (checkpoint / role).is_dir() or not any((checkpoint / role).iterdir()):
            raise ValueError(f'model role missing: {role}')
    data = _index(checkpoint / marker['library'])
    if marker['skill_count'] != len(data['skills']):
        raise ValueError('checkpoint/library skill count mismatch')
    return marker


def restore_library_pair(checkpoint, library, model_roles):
    """Restore a validated pair; retain the previous active directory as a backup."""
    checkpoint, library = Path(checkpoint), Path(library)
    marker = validate_library_pair(checkpoint, model_roles)
    if (library.resolve().is_relative_to(checkpoint.resolve())
            or checkpoint.resolve().is_relative_to(library.resolve())):
        raise ValueError('active library must not reside inside its checkpoint')
    library.parent.mkdir(parents=True, exist_ok=True)
    stage = library.with_name(library.name + '.restore_' + uuid4().hex)
    _copy_library(checkpoint / marker['library'], stage)
    backup = library.with_name(library.name + '.before_restore_' + uuid4().hex)
    existed = library.exists()
    if existed:
        os.replace(library, backup)
    try:
        os.replace(stage, library)
    except BaseException:
        if existed:
            os.replace(backup, library)
        raise
    return {'global_step': marker['global_step'], 'skill_count': marker['skill_count'],
            'previous_library_backup': str(backup) if existed else None}
