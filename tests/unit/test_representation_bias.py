"""Keep the CKA sensitivity audit signed and tied to its external reference."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit_representation_bias", Path(__file__).parents[2] / "scripts/audit_representation_bias.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_signed_sensitivity_and_ordinary_cka_are_separate():
    left = np.array([[0], [0], [1], [1]])
    right = np.array([[0], [1], [0], [1]])
    result = MODULE.compare_estimators(left, right, lambda *a, **k: -0.5)
    assert np.isclose(result["linear_cka"], 0)
    assert np.isclose(result["unbiased_hsic_cka"], -0.5)
    assert result["official_reference_absolute_error"] < 1e-12


@pytest.mark.parametrize("reference_value", [0.5, float("nan")])
def test_rejects_reference_mismatch(reference_value):
    values = np.arange(20).reshape(5, 4)
    with pytest.raises(ValueError, match="official reference"):
        MODULE.compare_estimators(values, values, lambda *a, **k: reference_value)


def test_untrusted_reference_is_rejected_before_parsing(tmp_path):
    source = tmp_path / "reference.ipynb"
    source.write_text("this is deliberately not a notebook")
    with pytest.raises(ValueError, match="checksum"):
        MODULE.official_cka(source)
