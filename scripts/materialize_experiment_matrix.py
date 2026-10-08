#!/usr/bin/env python3
"""Materialize a reviewed experiment matrix without duplicating hand-written YAML."""

from __future__ import annotations

import argparse

from invllava.config.matrix import materialize_experiment_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically materialize and validate a reviewed experiment matrix."
    )
    parser.add_argument("matrix")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-root", default="configs")
    args = parser.parse_args()
    output = materialize_experiment_matrix(
        args.matrix,
        args.output_dir,
        config_root=args.config_root,
    )
    print(output)


if __name__ == "__main__":
    main()
