import json
import numpy as np
import pytest
from vagen.skills.library import ViSkillLibrary


def test_numpy_positions_roundtrip_and_failed_save_preserves_index(tmp_path):
    library = ViSkillLibrary(None)
    library.admit({"features": [0.0], "trajectory": ["down"],
                   "positions": [(np.int64(0), np.int64(0)),
                                 (np.int64(1), np.int64(0))]})
    library.save(tmp_path)
    saved = json.loads((tmp_path / "skill_index.json").read_text())
    assert saved["skills"][0]["positions"] == [[0, 0], [1, 0]]
    restored = ViSkillLibrary(None)
    restored.load(tmp_path)
    assert restored._skills[0]["positions"] == [[0, 0], [1, 0]]
    library._skills[0]["positions"] = [(0.5, 0)]
    with pytest.raises(TypeError):
        library.save(tmp_path)
    assert json.loads((tmp_path / "skill_index.json").read_text()) == saved
