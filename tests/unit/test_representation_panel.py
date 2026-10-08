import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "panel", Path(__file__).parents[2] / "scripts/prepare_representation_panel.py"
)
PANEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PANEL)


def test_distinct_prompt_and_decoded_image_selection_is_order_independent():
    rows = [
        {"id": str(i), "prompt": p}
        for i, p in enumerate(
            ("What color?", " WHAT  color? ", "Which word?", "How many?", "Which animal?")
        )
    ]
    images = {str(i): str(i) for i in range(5)}
    images["3"] = images["2"]
    first = PANEL.select_rows(rows, images, count=3, seed=2026)
    assert first == PANEL.select_rows(rows[::-1], images, count=3, seed=2026)
    assert len({PANEL.normalized_prompt(r["prompt"]) for r in first}) == 3
    assert len({images[r["id"]] for r in first}) == 3
    with pytest.raises(ValueError, match="distinct"):
        PANEL.select_rows(rows, images, count=4, seed=2026)


def test_panel_rejects_duplicate_ids_and_invalid_size():
    row = {"id": "1", "prompt": "Question"}
    for rows, size in (([row, row], 1), ([row], 0)):
        with pytest.raises(ValueError, match="positive count"):
            PANEL.select_rows(rows, {"1": "image"}, count=size, seed=2026)
