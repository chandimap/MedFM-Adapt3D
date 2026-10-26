"""Explicit Commands for One Audited Representation and its Paired Random Control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import load_baseline_config
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.foundation.model import ARMS
from medfm_adapt3d.foundation.provenance import claim_gate, load_model_record
from medfm_adapt3d.foundation.training import compare_probes, run_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-record",
        type=Path,
        default=Path("configs/foundation/suprem_segresnet.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/baseline/random_segresnet.yaml"))
    run.add_argument("--index", type=Path, required=True)
    run.add_argument("--partitions", type=Path, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--phase", choices=("smoke", "calibration"), required=True)
    run.add_argument("--k", type=int, required=True)
    run.add_argument("--head-seed", type=int, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=2)
    compare = commands.add_parser("compare")
    compare.add_argument("--pretrained", type=Path, required=True)
    compare.add_argument("--random", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = load_model_record(args.model_record)
    if args.command == "audit":
        print(json.dumps(claim_gate(record), indent=2))
    elif args.command == "compare":
        compare_probes(args.pretrained, args.random, args.output)
        print(f"Completed descriptive validation comparison: {args.output}")
    else:
        if args.threads <= 0:
            raise ValueError("Thread count must be positive.")
        torch.set_num_threads(args.threads)
        protocol = LockedProtocol(
            CandidateIndex(args.index),
            load_baseline_config(args.config),
            args.partitions,
        )
        path = run_probe(
            protocol,
            record,
            arm=args.arm,
            k=args.k,
            head_seed=args.head_seed,
            phase=args.phase,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        print(f"Completed frozen-representation engineering run: {path}")
        print("Held-out test access: zero. Clean generalization and clinical claims: disabled.")


if __name__ == "__main__":
    main()
