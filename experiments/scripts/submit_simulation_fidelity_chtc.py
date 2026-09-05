"""Generate and optionally submit the distributed CHTC simulation-fidelity DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyrootutils

root = Path(pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True, dotenv=True))

DATASET_DEFAULTS = {
    "pancreas": {"config": "simulation_fidelity_chtc_pancreas", "title": "scIB Pancreas"},
    "immune": {"config": "simulation_fidelity_chtc_immune", "title": "scIB Immune"},
    "lung": {"config": "simulation_fidelity_chtc_lung", "title": "scIB Lung"},
    "wot": {"config": "simulation_fidelity_chtc_wot", "title": "Waddington-OT D0-D8"},
}
DEFAULT_DATASETS = ("pancreas", "immune", "lung")

METHOD_GROUPS = ("scdeepsim", "scvi", "scdiffusion", "scdesign3", "zinbwave")


@dataclass(frozen=True)
class Resources:
    cpus: int
    memory: str
    disk: str
    gpu: bool = False
    gpu_length: str | None = None


RESOURCE_MATRIX = {
    "scdeepsim": Resources(4, "24GB", "30GB", True, "short"),
    "scvi": Resources(4, "24GB", "30GB", True, "short"),
    "scdiffusion": Resources(4, "32GB", "40GB", True, "medium"),
    "scdesign3": Resources(4, "32GB", "30GB"),
    "zinbwave": Resources(2, "32GB", "30GB"),
    "aggregate": Resources(4, "16GB", "20GB"),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transfer_basename(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path if parsed.scheme else value
    name = Path(path).name
    if not name:
        raise ValueError(f"Cannot derive a transferred filename from {value!r}.")
    return name


def ensure_osdf_parent(value: str) -> None:
    """Create a mounted CHTC staging parent when running on a submit node."""
    prefix = "osdf:///chtc/staging/"
    if not value.startswith(prefix):
        return
    local_path = Path("/staging") / value.removeprefix(prefix)
    if Path("/staging").is_dir():
        local_path.parent.mkdir(parents=True, exist_ok=True)


def _submit_header(initial_dir: Path, resources: Resources, image: str) -> list[str]:
    lines = [
        "universe = vanilla",
        f"initialdir = {initial_dir}",
        f"container_image = {image}",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        f"request_cpus = {resources.cpus}",
        f"request_memory = {resources.memory}",
        f"request_disk = {resources.disk}",
        "log = job.log",
        "output = job.out",
        "error = job.err",
    ]
    if resources.gpu:
        lines.extend(
            [
                "request_gpus = 1",
                "+WantGPULab = true",
                f'+GPUJobLength = "{resources.gpu_length}"',
                "requirements = (CUDAGlobalMemoryMb >= 16000)",
            ]
        )
    return lines


def _condor_arguments(parts: list[str]) -> str:
    """Encode argv using HTCondor's new arguments syntax."""
    encoded = []
    for part in map(str, parts):
        escaped = part.replace("'", "''").replace('"', '""')
        if not escaped or any(char.isspace() for char in escaped) or "'" in part:
            escaped = f"'{escaped}'"
        encoded.append(escaped)
    return f'arguments = "{" ".join(encoded)}"'


def _write_parent_submit(
    *,
    path: Path,
    run_script: Path,
    source_bundle: Path,
    data_transfer: str,
    data_basename: str,
    image_transfer: str,
    config_name: str,
    method_group: str,
    checksum: str,
    smoke: bool,
    artifact_transfer: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_name = f"output_{method_group}"
    lines = _submit_header(path.parent, RESOURCE_MATRIX[method_group], image_transfer)
    lines.extend(
        [
            "executable = /bin/bash",
            f"transfer_input_files = {run_script}, {source_bundle}, {data_transfer}",
            _condor_arguments(
                [
                    run_script.name,
                    source_bundle.name,
                    data_basename,
                    config_name,
                    method_group,
                    checksum,
                    output_name,
                    "1" if smoke else "0",
                ]
            ),
            f"transfer_output_files = {method_group}.tar.gz",
            "transfer_output_remaps = "
            f'"{method_group}.tar.gz = {artifact_transfer}"',
            "queue",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_aggregate_submit(
    *,
    path: Path,
    run_script: Path,
    source_bundle: Path,
    data_transfer: str,
    data_basename: str,
    image_transfer: str,
    config_name: str,
    dataset_id: str,
    dataset_title: str,
    checksum: str,
    parent_transfers: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _submit_header(path.parent, RESOURCE_MATRIX["aggregate"], image_transfer)
    inputs = [str(run_script), str(source_bundle), data_transfer, *parent_transfers]
    lines.extend(
        [
            "executable = /bin/bash",
            f"transfer_input_files = {', '.join(inputs)}",
            _condor_arguments(
                [
                    run_script.name,
                    source_bundle.name,
                    data_basename,
                    config_name,
                    dataset_id,
                    dataset_title,
                    checksum,
                ]
            ),
            "transfer_output_files = official",
            "queue",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def generate_workflow(
    *,
    datasets: list[str],
    seed: int,
    asset_manifest_path: Path,
    source_bundle: Path,
    batch_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Generate submit descriptions and a strict DAG without submitting it."""
    if seed != 42:
        raise ValueError("This benchmark is locked to seed 42.")
    unknown = sorted(set(datasets) - set(DATASET_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}")
    assets = json.loads(asset_manifest_path.read_text())
    image = assets["container"]
    if not image.get("transfer", "").startswith("osdf:///"):
        raise ValueError("The formal container must be addressed through osdf:///.")
    run_output_root = str(
        assets.get(
            "run_output_root",
            "osdf:///chtc/staging/l/lma229/scdeepsim-simulation-fidelity/runs",
        )
    ).rstrip("/")
    if not run_output_root.startswith("osdf:///"):
        raise ValueError("Run artifacts must be written through osdf:///.")

    batch_dir.mkdir(parents=True, exist_ok=True)
    run_method = root / "chtc/simulation_fidelity/run_method.sh"
    run_aggregate = root / "chtc/simulation_fidelity/run_aggregate.sh"
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dag_lines: list[str] = []
    node_count = 0
    dataset_records: dict[str, Any] = {}

    for dataset_id in datasets:
        defaults = DATASET_DEFAULTS[dataset_id]
        asset = assets["datasets"][dataset_id]
        data_transfer = str(asset["transfer"])
        data_basename = transfer_basename(data_transfer)
        checksum = str(asset["checksum"])
        parent_names = []
        parent_transfers = []
        for group in METHOD_GROUPS:
            node_name = f"{dataset_id}_{group}".upper()
            node_dir = batch_dir / "nodes" / dataset_id / group
            submit_path = node_dir / "method.sub"
            artifact_transfer = (
                f"{run_output_root}/{batch_dir.name}/{dataset_id}/{group}.tar.gz"
            )
            ensure_osdf_parent(artifact_transfer)
            _write_parent_submit(
                path=submit_path,
                run_script=run_method,
                source_bundle=source_bundle,
                data_transfer=data_transfer,
                data_basename=data_basename,
                image_transfer=str(image["transfer"]),
                config_name=str(defaults["config"]),
                method_group=group,
                checksum=checksum,
                smoke=smoke,
                artifact_transfer=artifact_transfer,
            )
            dag_lines.extend([f"JOB {node_name} {submit_path}", f"RETRY {node_name} 2"])
            parent_names.append(node_name)
            parent_transfers.append(artifact_transfer)
            node_count += 1

        aggregate_name = f"{dataset_id}_aggregate".upper()
        aggregate_submit = batch_dir / "nodes" / dataset_id / "aggregate" / "aggregate.sub"
        _write_aggregate_submit(
            path=aggregate_submit,
            run_script=run_aggregate,
            source_bundle=source_bundle,
            data_transfer=data_transfer,
            data_basename=data_basename,
            image_transfer=str(image["transfer"]),
            config_name=str(defaults["config"]),
            dataset_id=dataset_id,
            dataset_title=str(defaults["title"]),
            checksum=checksum,
            parent_transfers=parent_transfers,
        )
        dag_lines.extend(
            [
                f"JOB {aggregate_name} {aggregate_submit}",
                f"PARENT {' '.join(parent_names)} CHILD {aggregate_name}",
            ]
        )
        node_count += 1
        dataset_records[dataset_id] = {
            **defaults,
            "asset": asset,
            "parent_nodes": parent_names,
            "aggregate_node": aggregate_name,
            "parent_artifacts": parent_transfers,
        }

    dag_path = batch_dir / "simulation_fidelity.dag"
    dag_path.write_text("\n".join(dag_lines) + "\n")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "smoke": smoke,
        "source_commit": source_commit,
        "source_bundle": str(source_bundle),
        "source_bundle_sha256": sha256_file(source_bundle),
        "container": image,
        "run_output_root": run_output_root,
        "datasets": dataset_records,
        "formal_node_count": node_count,
        "dag": str(dag_path),
        "resource_matrix": {
            name: resources.__dict__ for name, resources in RESOURCE_MATRIX.items()
        },
    }
    (batch_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DEFAULTS),
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_workflow(
        datasets=args.datasets,
        seed=args.seed,
        asset_manifest_path=args.asset_manifest.expanduser().resolve(),
        source_bundle=args.source_bundle.expanduser().resolve(),
        batch_dir=args.batch_dir.expanduser().resolve(),
        smoke=args.smoke,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.submit:
        subprocess.run(
            ["condor_submit_dag", "-force", manifest["dag"]],
            check=True,
            cwd=args.batch_dir,
        )


if __name__ == "__main__":
    main()
