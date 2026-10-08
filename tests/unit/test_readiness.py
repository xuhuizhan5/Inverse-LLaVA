from invllava.reproducibility import audit_reproduction


def test_repository_code_stage_is_ready() -> None:
    report = audit_reproduction("configs/reproduction/training_local.yaml", stage="code")
    assert report.passed
    assert all(check.status == "ready" for check in report.checks)


def test_repository_scientific_stage_reports_known_blockers() -> None:
    report = audit_reproduction("configs/reproduction/training_local.yaml", stage="scientific")
    assert not report.passed
    blocked = {check.id for check in report.checks if check.status == "blocked"}
    assert "dependency-lock" not in blocked
    assert "base-image-lock" not in blocked
    assert "publication-image-lock" in blocked
    assert "experiment:canonical_7b:provenance" not in blocked
    assert "benchmark:textvqa:provenance" not in blocked


def test_external_gate_does_not_block_training_gate() -> None:
    training = audit_reproduction("configs/reproduction/training_local.yaml", stage="scientific")
    external = audit_reproduction("configs/reproduction/external_final.yaml", stage="scientific")

    training_blocked = {check.id for check in training.checks if check.status == "blocked"}
    external_blocked = {check.id for check in external.checks if check.status == "blocked"}
    assert "benchmark:mmvet_gpt41_hosted:provenance" not in training_blocked
    assert "benchmark:vqav2_testdev:provenance" not in training_blocked
    assert "benchmark:mmvet_gpt41_hosted:provenance" in external_blocked
    assert "benchmark:vqav2_testdev:provenance" in external_blocked


def test_full_reproduction_selects_the_primary_evaluation_protocols() -> None:
    report = audit_reproduction("configs/reproduction/full.yaml", stage="code")
    assert report.passed
    checks = {check.id for check in report.checks}
    assert {
        "benchmark:mmbench_en_llava",
        "benchmark:mmbench_cn_llava",
        "benchmark:mmvet_gpt41_hosted",
        "benchmark:vqav2_testdev",
    } <= checks
    assert "benchmark:mmvet" not in checks
