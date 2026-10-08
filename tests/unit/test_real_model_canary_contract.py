from __future__ import annotations

import re
from pathlib import Path


def test_real_model_training_downloads_use_both_authorization_gates() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "scripts/run_real_model_canary.sh").read_text(encoding="utf-8")
    launches = re.findall(
        r'"\$python_bin" scripts/launch_train\.py.*?(?=\n\n|\Z)',
        script,
        flags=re.DOTALL,
    )

    assert len(launches) == 2
    assert all("--allow-download" in launch for launch in launches)
    assert "INVLLAVA_ALLOW_DOWNLOADS:-0" in script
    assert "fusion.0.q.mapped_text" in script
    assert "fusion.0.mapped_text" not in script
