"""CPU scientific contract gate; synthetic evidence, never real LIDC performance."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    fingerprint,
    implementation_fingerprint,
    seal,
    write_json,
)
from medfm_adapt3d.baseline.candidates import INPUT_TRACE_FIELDS, CandidateIndex
from medfm_adapt3d.baseline.config import SmokeBudget, load_baseline_config
from medfm_adapt3d.baseline.metrics import (
    ObservationMetrics,
    aggregate_patients,
    segmentation_metrics,
)
from medfm_adapt3d.baseline.model import build_random_model
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition, expected_partition
from medfm_adapt3d.baseline.reporting import read_completed_run
from medfm_adapt3d.baseline.sampling import PatientBalancedSampler, augment_pair
from medfm_adapt3d.baseline.selection import (
    CheckpointSelector,
    RunAccess,
    require_optimization_adequacy,
)
from medfm_adapt3d.baseline.synthetic import create_synthetic_candidates
from medfm_adapt3d.baseline.training import optimization_diagnostics, run_baseline


def validate(root: Path) -> list[str]:
    checks = []

    def require(name: str, condition: bool) -> None:
        if not condition:
            raise RuntimeError(f"FAIL: {name}")
        checks.append(name)
        print(f"PASS: {name}", flush=True)

    torch.set_num_threads(2)
    cfg = load_baseline_config(Path("configs/baseline/random_segresnet.yaml"))
    index = CandidateIndex(create_synthetic_candidates(root / "candidates", patients=34))
    path = root / "partitions.json"
    create_partition(index, cfg, path)
    protocol = LockedProtocol(index, cfg, path)
    groups = [set(protocol.payload["patients"][name]) for name in ("train", "validation", "test")]
    require(
        "patient-disjoint partitions",
        not any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)),
    )
    supports = [set(protocol.payload["supports"][str(k)]["patient_ids"]) for k in (1, 5, 10, 20)]
    require(
        "nested 1/5/10/20 patient budgets",
        all(len(s) == k for s, k in zip(supports, (1, 5, 10, 20), strict=True))
        and all(a < b for a, b in zip(supports[:-1], supports[1:], strict=True)),
    )
    changed = replace(cfg, seeds=replace(cfg.seeds, model=(2, 3, 5, 7, 11)))
    require(
        "partition and support locking independent of model seed",
        expected_partition(index, cfg) == expected_partition(index, changed),
    )
    require(
        "shared-recipe calibration covers feasible extremes",
        protocol.calibration_budgets == (1, 20),
    )
    fingerprints, full_training = [], True
    for seed in cfg.seeds.model:
        _, audit = build_random_model(cfg.model, seed)
        fingerprints.append(audit["initial_state_sha256"])
        full_training &= (
            audit["fraction_trainable"] == 1 and not audit["weights_loaded_during_initialization"]
        )
    require("five independent random initializations", len(set(fingerprints)) == 5)
    require("all intended parameters trainable; no pretrained loading", full_training)
    sampler = PatientBalancedSampler(
        [index.by_id[i] for i in protocol.support_ids(20)],
        cfg.seeds.loader,
    )
    for _ in range(80):
        sampler.next_id()
    require(
        "patient-balanced sampling and complete reader rotation",
        set(sampler.patient_exposure.values()) == {4}
        and all(n > 0 for n in sampler.target_exposure.values()),
    )
    require(
        "target-metadata isolation with explicitly declared oracle centre",
        all(
            set(r.trace) == set(INPUT_TRACE_FIELDS) and "mask_geometry" not in r.trace
            for r in index.records
        ),
    )
    for identifier in protocol.partition_ids("test"):
        index.verify_integrity(identifier)  # Opaque hashes, no target/QC parsing.
    require("opaque target/QC integrity without shape disclosure", True)
    generator = torch.Generator().manual_seed(cfg.seeds.augmentation)
    rng_state = generator.get_state().clone()
    image = torch.arange(27).reshape(1, 3, 3, 3).float()
    target = (image > 10).to(torch.uint8)
    views = augment_pair(
        image, target, generator=generator, probability=cfg.sampling.flip_probability
    )
    require(
        "conservative identity augmentation and independent RNG",
        torch.equal(views[0], image)
        and torch.equal(views[1], target)
        and torch.equal(rng_state, generator.get_state()),
    )
    rows = [ObservationMetrics("patient_a", str(i), 1, 0, 0, 1, False, 1, 1) for i in range(4)]
    rows.append(ObservationMetrics("patient_b", "b", 0, None, 1, 0, True, 1, 0))
    macro = aggregate_patients(rows)
    require(
        "patient-level aggregation and reader dependence", macro["patient_macro"]["dice"] == 0.5
    )
    reference = np.zeros((7, 7, 7), dtype=np.uint8)
    reference[3, 3, 3] = 1
    metric = segmentation_metrics(
        np.roll(reference, 1, axis=0),
        reference,
        spacing_mm_xyz=(2, 3, 4),
        patient_id="phantom",
        candidate_id="physical",
    )
    require("HD95 in actual physical units", metric.hd95_mm == 4)
    empty = segmentation_metrics(
        np.zeros_like(reference),
        reference,
        spacing_mm_xyz=(2, 3, 4),
        patient_id="phantom",
        candidate_id="empty",
    )
    require(
        "physical volume error and honest empty predictions",
        empty.absolute_volume_error_mm3 == 24 and empty.hd95_mm is None and empty.dice == 0,
    )
    access = RunAccess(protocol, 1, "calibration")
    isolated = False
    try:
        access.read("test", access.allowed["test"][0])
    except RuntimeError:
        isolated = True
    require("test labels isolated from training/calibration", isolated)
    selector = CheckpointSelector(cfg.optimization)
    rejected = False
    try:
        selector.observe(step=200, patient_macro_dice=1, source="test")
    except ValueError:
        rejected = True
    require("checkpoint selection accepts validation only", rejected)
    # Analytic adversarial trajectory around the actual state-machine contract.
    # These scores are test inputs, not medical/model-performance measurements.
    improving = CheckpointSelector(cfg.optimization)
    for step in range(200, cfg.optimization.max_steps + 1, 200):
        improving.observe(step=step, patient_macro_dice=step / 10000, source="validation")
    validation_fixture = {
        "counts": {"reader_targets": 1, "empty_predictions": 0},
        "patient_macro": {"dice": 0.5},
        "observations": [{"candidate_id": "fixture", "predicted_voxels": 1}],
    }
    diagnostics = optimization_diagnostics(
        improving,
        actual_steps=cfg.optimization.max_steps,
        stopping_reason="maximum_steps",
        validation=validation_fixture,
        patient_exposure={"fixture": 1},
        target_exposure={"fixture": 1},
        candidate_sizes={"fixture": 27},
    )
    rejected = False
    try:
        require_optimization_adequacy(diagnostics)
    except RuntimeError:
        rejected = True
    require("optimization adequacy rejects continuing improvement at cap", rejected)
    ready = RunAccess(protocol, 1, "definitive")
    checkpoint = root / "state-machine-checkpoint.pt"
    checkpoint.write_bytes(b"Synthetic state-machine marker, not trained model weights.")
    lock = {
        "checkpoint_sha256": file_hash(checkpoint),
        "partition_sha256": protocol.sha256,
        "candidate_index_sha256": index.sha256,
        "config_sha256": fingerprint(cfg.to_dict()),
        "implementation_sha256": implementation_fingerprint(),
        "selected_step": 200,
        "selection_source": "validation",
    }
    lock_path = root / "state-machine-lock.json"
    write_json(lock_path, seal(lock))
    ready.lock(lock_path, checkpoint)
    for identifier in ready.allowed["validation"]:
        ready.read("validation", identifier)
    ready.final_validation_completed()
    blocked_gate = root / "blocked-gate.json"
    write_json(
        blocked_gate,
        seal(
            {
                **lock,
                "diagnostics": diagnostics,
                "actual_steps": cfg.optimization.max_steps,
                "test_label_reads": 0,
                "status": "blocked",
            }
        ),
    )
    try:
        ready.approve_optimization(blocked_gate)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("FAIL: improving definitive run opened test.")
    require(
        "held-out test firewall stays closed after failed adequacy",
        ready.read_counts["test"] == 0 and ready.state == "validated",
    )
    plateau = CheckpointSelector(cfg.optimization)
    for step in range(200, 2401, 200):
        plateau.observe(step=step, patient_macro_dice=0.5, source="validation")
    adequate = optimization_diagnostics(
        plateau,
        actual_steps=2400,
        stopping_reason="validation_early_stopping",
        validation=validation_fixture,
        patient_exposure={"fixture": 1},
        target_exposure={"fixture": 1},
        candidate_sizes={"fixture": 27},
    )
    passed_gate = root / "passed-gate.json"
    write_json(
        passed_gate,
        seal(
            {
                **lock,
                "diagnostics": adequate,
                "actual_steps": 2400,
                "test_label_reads": 0,
                "status": "passed",
            }
        ),
    )
    ready.approve_optimization(passed_gate)
    ready.open_test()
    for identifier in ready.allowed["test"]:
        ready.read("test", identifier)
    ready.finish_test()
    require(
        "adequacy precedes locked one-time target/QC evaluation",
        ready.state == "complete" and ready.read_counts["test"] == len(ready.allowed["test"]),
    )
    short_cfg = replace(cfg, smoke=SmokeBudget(2, 1, 1, 1))
    short_protocol = LockedProtocol(index, short_cfg, path)
    run = run_baseline(
        short_protocol,
        k=1,
        model_seed=short_cfg.seeds.smoke,
        phase="smoke",
        output_root=root / "runs",
    )
    result, provenance = read_completed_run(run)
    events = [e["event"] for e in result["test_access"]["events"]]
    require(
        "checkpoint lock precedes final validation and held-out test",
        events
        == [
            "checkpoint_locked",
            "final_validation_completed",
            "test_evaluation_started",
            "test_evaluation_completed",
        ],
    )
    require(
        "provenance schema and result artifacts",
        all(
            key in provenance
            for key in (
                "runtime",
                "git_dirty",
                "model",
                "seeds",
                "implementation_sha256",
                "config_sha256",
                "preprocessing_sha256",
                "patient_assignments",
                "support_patients",
            )
        )
        and result["actual_optimization_steps"] >= 1,
    )
    immutable = False
    try:
        run_baseline(
            short_protocol,
            k=1,
            model_seed=short_cfg.seeds.smoke,
            phase="smoke",
            output_root=root / "runs",
        )
    except FileExistsError:
        immutable = True
    require("immutable completed output", immutable)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional exclusive JSON evidence file.")
    args = parser.parse_args()
    try:
        with TemporaryDirectory(prefix="medfm-baseline-contract-") as temporary:
            checks = validate(Path(temporary))
    except Exception as error:
        print(f"MedFM-Adapt3D random baseline contract: FAIL ({error})", flush=True)
        raise
    if args.report:
        write_json(
            args.report,
            {
                "status": "PASS",
                "checks": checks,
                "evidence_source": "synthetic",
                "clinical_performance_claimed": False,
            },
        )
    print("MedFM-Adapt3D random baseline contract: PASS")
    print(
        "Evidence: synthetic/adversarial checks; no real LIDC performance or population inference."
    )


if __name__ == "__main__":
    main()
