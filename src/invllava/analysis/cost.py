from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostEstimate:
    gpu_count: int
    hourly_price_per_gpu: float
    wall_hours: float
    gpu_hours: float
    compute_cost: float
    fixed_cost: float
    contingency_fraction: float
    contingency_cost: float
    estimated_total: float
    budget: float | None
    budget_remaining: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def project_wall_hours(
    *, measured_wall_hours: float, measured_examples: int, target_examples: int
) -> float:
    """Linear first-pass projection; callers must replace it with measured full-run time."""

    if measured_wall_hours <= 0 or measured_examples <= 0 or target_examples <= 0:
        raise ValueError("calibration hours and example counts must be positive")
    return measured_wall_hours * target_examples / measured_examples


def estimate_cost(
    *,
    gpu_count: int,
    hourly_price_per_gpu: float,
    wall_hours: float,
    fixed_cost: float = 0.0,
    contingency_fraction: float = 0.1,
    budget: float | None = None,
) -> CostEstimate:
    if gpu_count <= 0 or hourly_price_per_gpu <= 0 or wall_hours <= 0:
        raise ValueError("GPU count, hourly price, and wall hours must be positive")
    if fixed_cost < 0 or contingency_fraction < 0 or (budget is not None and budget < 0):
        raise ValueError("fixed cost, contingency, and budget must be non-negative")
    gpu_hours = gpu_count * wall_hours
    compute = gpu_hours * hourly_price_per_gpu
    subtotal = compute + fixed_cost
    contingency = subtotal * contingency_fraction
    total = subtotal + contingency
    return CostEstimate(
        gpu_count=gpu_count,
        hourly_price_per_gpu=hourly_price_per_gpu,
        wall_hours=wall_hours,
        gpu_hours=gpu_hours,
        compute_cost=compute,
        fixed_cost=fixed_cost,
        contingency_fraction=contingency_fraction,
        contingency_cost=contingency,
        estimated_total=total,
        budget=budget,
        budget_remaining=None if budget is None else budget - total,
    )
