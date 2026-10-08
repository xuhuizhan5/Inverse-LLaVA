from invllava.analysis.cost import estimate_cost, project_wall_hours


def test_cost_estimate_keeps_wall_time_and_gpu_hours_distinct() -> None:
    result = estimate_cost(
        gpu_count=8,
        hourly_price_per_gpu=3.0,
        wall_hours=2.0,
        fixed_cost=10.0,
        contingency_fraction=0.1,
        budget=100.0,
    )
    assert result.gpu_hours == 16.0
    assert result.compute_cost == 48.0
    assert abs(result.estimated_total - 63.8) < 1e-9
    assert result.budget_remaining is not None
    assert abs(result.budget_remaining - 36.2) < 1e-9


def test_linear_projection_is_explicit() -> None:
    assert (
        project_wall_hours(measured_wall_hours=0.5, measured_examples=100, target_examples=1000)
        == 5.0
    )
