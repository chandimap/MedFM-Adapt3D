"""Standalone CPU end-to-end baseline verification without private medical data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import load_baseline_config
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition
from medfm_adapt3d.baseline.reporting import read_completed_run
from medfm_adapt3d.baseline.synthetic import create_synthetic_candidates
from medfm_adapt3d.baseline.training import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/random_baseline_synthetic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = load_baseline_config(Path("configs/baseline/random_segresnet.yaml"))
    index = CandidateIndex(create_synthetic_candidates(args.output / "candidates", patients=10))
    partitions = args.output / "partitions.json"
    create_partition(index, config, partitions)
    protocol = LockedProtocol(index, config, partitions)
    run = run_baseline(
        protocol,
        k=1,
        model_seed=config.seeds.smoke,
        phase="smoke",
        output_root=args.output / "runs",
    )
    result, _ = read_completed_run(run)
    if result["test"] is None or result["test_access"]["read_counts"]["test"] <= 0:
        raise RuntimeError("Synthetic smoke did not exercise final held-out evaluation.")
    print("Random baseline synthetic end-to-end: PASS")
    print(f"Actual optimizer steps: {result['actual_optimization_steps']}")
    print(f"Selected validation checkpoint step: {result['selected_step']}")
    print(f"Inspectable synthetic artifacts: {run}")
    print("Synthetic software evidence only; no real LIDC performance is claimed.")


if __name__ == "__main__":
    main()
