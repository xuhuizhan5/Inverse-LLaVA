#!/usr/bin/env python3
"""Estimate an experiment job from a current user-entered cloud price."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from invllava.analysis.cost import estimate_cost, project_wall_hours
from invllava.artifacts.atomic import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu-count", required=True, type=int)
    parser.add_argument("--hourly-price-per-gpu", required=True, type=float)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--wall-hours", type=float)
    duration.add_argument("--measured-wall-hours", type=float)
    parser.add_argument("--measured-examples", type=int)
    parser.add_argument("--target-examples", type=int)
    parser.add_argument("--fixed-cost", type=float, default=0.0)
    parser.add_argument("--contingency-fraction", type=float, default=0.1)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--price-observed-at", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.wall_hours is not None:
        wall_hours = args.wall_hours
        projection = "direct"
    else:
        if args.measured_examples is None or args.target_examples is None:
            parser.error("--measured-wall-hours requires --measured-examples and --target-examples")
        wall_hours = project_wall_hours(
            measured_wall_hours=args.measured_wall_hours,
            measured_examples=args.measured_examples,
            target_examples=args.target_examples,
        )
        projection = "linear_from_calibration"
    estimate = estimate_cost(
        gpu_count=args.gpu_count,
        hourly_price_per_gpu=args.hourly_price_per_gpu,
        wall_hours=wall_hours,
        fixed_cost=args.fixed_cost,
        contingency_fraction=args.contingency_fraction,
        budget=args.budget,
    )
    payload = {
        "name": args.name,
        "calculation": estimate.to_dict(),
        "duration_basis": projection,
        "price_source": args.price_source,
        "price_observed_at": args.price_observed_at,
        "estimated_at": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "A linear screen-to-full projection is provisional; replace it after calibration."
            if projection == "linear_from_calibration"
            else "The direct estimate bills every reserved GPU for the entered wall time."
        ),
    }
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        atomic_write_json(destination, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
