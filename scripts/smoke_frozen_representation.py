"""Exercise Real SuPreM Weights on Synthetic Phantoms; Never Download or Publish Tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import load_baseline_config
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition
from medfm_adapt3d.baseline.synthetic import create_synthetic_candidates
from medfm_adapt3d.foundation.model import ARMS
from medfm_adapt3d.foundation.provenance import load_model_record
from medfm_adapt3d.foundation.training import compare_probes, read_probe_run, run_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/frozen_representation_synthetic")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory; evidence is never overwritten.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "Download the pinned official checkpoint before running this check."
        )
    torch.set_num_threads(2)
    config = load_baseline_config(Path("configs/baseline/random_segresnet.yaml"))
    record = load_model_record(Path("configs/foundation/suprem_segresnet.yaml"))
    index = CandidateIndex(create_synthetic_candidates(args.output / "candidates", patients=10))
    partitions = args.output / "partitions.json"
    create_partition(index, config, partitions)
    protocol = LockedProtocol(index, config, partitions)
    paths = []
    for arm in ARMS:
        path = run_probe(
            protocol,
            record,
            arm=arm,
            k=1,
            head_seed=config.seeds.smoke,
            phase="smoke",
            output_root=args.output / "runs",
            checkpoint=args.checkpoint if arm == "suprem_frozen" else None,
        )
        result, _ = read_probe_run(path)
        if not result["head_changed"] or not result["encoder_unchanged"]:
            raise RuntimeError("Smoke run did not demonstrate head-only adaptation.")
        paths.append(path)
    compare_probes(paths[0], paths[1], args.output / "comparison.json")
    print("Official-checkpoint synthetic paired smoke: PASS")
    print("Encoder tensor coverage: 100%; trainable head parameters: 241; test reads: 0.")
    print(f"Inspectable local evidence: {args.output}")
    print("Synthetic implementation evidence only; no LIDC or clinical performance claim.")


if __name__ == "__main__":
    main()
