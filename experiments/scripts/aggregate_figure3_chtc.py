"""Aggregate five CHTC parent jobs into two strict Figure 3 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=(".git", "AGENTS.md"), pythonpath=True, dotenv=True
)

from experiments.src.figure3_quality.aggregate import aggregate_and_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-title", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--data-checksum", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_and_plot(
        project_root=Path(root),
        parent_root=args.parent_root,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        dataset_title=args.dataset_title,
        config_name=args.config_name,
        data_path=args.data_path,
        data_checksum=args.data_checksum,
    )


if __name__ == "__main__":
    main()
