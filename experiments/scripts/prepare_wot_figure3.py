"""Create the D0-D8 Waddington-OT input used by Figure 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True, dotenv=True)

from experiments.src.figure3_quality.wot import prepare_wot_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--cell-days", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_wot_files(
        args.expression,
        args.cell_days,
        args.batches,
        args.output,
    )
    rendered = json.dumps(metadata, indent=2, sort_keys=True)
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
