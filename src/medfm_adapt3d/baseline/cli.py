"""Explicit Local Preparation, Partition Lock, Calibration, Freeze, and Run Commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from medfm_adapt3d.baseline.candidates import CandidateIndex, prepare_lidc_candidates
from medfm_adapt3d.baseline.config import load_baseline_config
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition
from medfm_adapt3d.baseline.reporting import freeze_recipe, summarize_runs
from medfm_adapt3d.baseline.training import run_baseline, verify_frozen_recipe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare private candidates through existing QC.")
    prepare.add_argument("--dicom-root", type=Path, required=True)
    prepare.add_argument("--xml-root", type=Path, required=True)
    prepare.add_argument(
        "--preprocessing", type=Path, default=Path("configs/preprocessing/lidc_candidate.yaml")
    )
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("partition", "inspect", "run", "matrix", "freeze", "summarize"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config", type=Path, default=Path("configs/baseline/random_segresnet.yaml")
        )
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--partitions", type=Path, required=True)
        if name in ("run", "matrix"):
            command.add_argument("--output-root", type=Path, required=True)
            command.add_argument("--device", default="cpu")
            command.add_argument("--threads", type=int, default=2)
            command.add_argument(
                "--k", type=int, nargs="+" if name == "matrix" else None, required=True
            )
            command.add_argument("--frozen-recipe", type=Path)
        if name == "run":
            command.add_argument(
                "--phase", choices=("smoke", "calibration", "definitive"), required=True
            )
            command.add_argument("--model-seed", type=int, required=True)
        if name == "freeze":
            command.add_argument("--calibration-run", type=Path, action="append", required=True)
            command.add_argument("--rationale", required=True)
            command.add_argument("--output", type=Path, required=True)
        if name == "summarize":
            command.add_argument("--output-root", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--frozen-recipe", type=Path, required=True)
            command.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        path = prepare_lidc_candidates(
            dicom_root=args.dicom_root,
            xml_root=args.xml_root,
            preprocessing_path=args.preprocessing,
            output=args.output,
        )
        index = CandidateIndex(path)
        print(
            f"PASS: {len(index.patients)} eligible patients, {len(index.records)} reader targets."
        )
        print(f"Private candidate index: {path}")
        return
    config = load_baseline_config(args.config)
    index = CandidateIndex(args.index)
    if args.command == "partition":
        create_partition(index, config, args.partitions)
    protocol = LockedProtocol(index, config, args.partitions)
    if args.command in ("partition", "inspect"):
        for name, patients in protocol.payload["patients"].items():
            print(f"{name}: {len(patients)} eligible patients")
        print(f"Feasible patient K: {list(map(int, protocol.payload['supports']))}")
        print(f"Shared-recipe calibration K: {list(protocol.calibration_budgets)}")
        print(
            "Unavailable K (explicit failures if requested): "
            f"{protocol.payload['unavailable_budgets']}"
        )
        print("PASS: patient-disjoint partitions and deterministic nested support locked.")
        return
    if args.command == "freeze":
        freeze_recipe(protocol, args.calibration_run, output=args.output, rationale=args.rationale)
        print(f"PASS: train/validation-only recipe frozen: {args.output}")
        return
    if args.command == "summarize":
        summarize_runs(
            protocol,
            output_root=args.output_root,
            output=args.output,
            frozen_recipe=args.frozen_recipe,
            allow_incomplete=args.allow_incomplete,
        )
        print(f"PASS: all-seed summaries and label-efficiency CSV: {args.output}")
        return
    if args.threads <= 0:
        raise ValueError("--threads must be a positive integer.")
    torch.set_num_threads(args.threads)
    if args.command == "matrix":
        if args.frozen_recipe is None:
            raise ValueError("A definitive matrix requires --frozen-recipe.")
        # Preflight ALL budgets before starting any run: never silently reduce K.
        if len(set(args.k)) != len(args.k):
            raise ValueError("Duplicate K budgets are not allowed.")
        for k in args.k:
            protocol.support_ids(k)
        verify_frozen_recipe(args.frozen_recipe, protocol)
        if index.payload["synthetic"]:
            raise ValueError("Definitive matrices require real-data candidates.")
        for k in args.k:
            for seed in config.seeds.model:
                path = args.output_root / "definitive" / f"k{k:02d}" / f"seed{seed:03d}"
                if path.exists():
                    raise FileExistsError(
                        f"Matrix output already exists: {path}; use a new output root."
                    )
        for k in args.k:
            for seed in config.seeds.model:
                path = run_baseline(
                    protocol,
                    k=k,
                    model_seed=seed,
                    phase="definitive",
                    output_root=args.output_root,
                    device=args.device,
                    frozen_recipe=args.frozen_recipe,
                )
                print(f"PASS: completed K={k}, seed={seed}: {path}", flush=True)
        return
    path = run_baseline(
        protocol,
        k=args.k,
        model_seed=args.model_seed,
        phase=args.phase,
        output_root=args.output_root,
        device=args.device,
        frozen_recipe=args.frozen_recipe,
    )
    print(f"PASS: completed {args.phase} run: {path}")
    print("Smoke/calibration establish software/optimization evidence, not definitive performance.")


if __name__ == "__main__":
    main()
