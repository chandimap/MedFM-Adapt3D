import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from medfm_adapt3d.baseline import training
from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    fingerprint,
    implementation_fingerprint,
    read_sealed,
    seal,
    write_json,
)
from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import BaselineConfig, SmokeBudget
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition
from medfm_adapt3d.baseline.reporting import freeze_recipe, read_completed_run
from medfm_adapt3d.baseline.selection import (
    CheckpointSelector,
    RunAccess,
    require_optimization_adequacy,
)
from medfm_adapt3d.baseline.training import release_test, run_baseline, verify_frozen_recipe


@pytest.fixture
def private_index(candidate_index: CandidateIndex, tmp_path: Path) -> CandidateIndex:
    root = tmp_path / "private"
    shutil.copytree(candidate_index.root, root)
    return CandidateIndex(root / "index.json")


def _protocol(index: CandidateIndex, config: BaselineConfig, path: Path) -> LockedProtocol:
    create_partition(index, config, path)
    return LockedProtocol(index, config, path)


def test_corrupted_candidate_array_fails(private_index: CandidateIndex) -> None:
    record = private_index.records[0]
    path = private_index.root / record.image_path
    path.write_bytes(b"corrupted bytes")
    with pytest.raises(ValueError, match="Candidate file integrity"):
        private_index.load(record.candidate_id)


def test_only_validation_selects_checkpoint_and_ties_keep_earliest(
    baseline_config: BaselineConfig,
) -> None:
    selector = CheckpointSelector(baseline_config.optimization)
    with pytest.raises(ValueError, match="Only validation"):
        selector.observe(step=200, patient_macro_dice=0.99, source="test")
    assert selector.observe(step=200, patient_macro_dice=0.6, source="validation")
    assert not selector.observe(step=400, patient_macro_dice=0.6, source="validation")
    assert selector.observe(step=600, patient_macro_dice=0.6001, source="validation")
    assert selector.best_step == 600  # min_delta is not a checkpoint selection penalty.
    assert not selector.should_stop(600)


def test_early_stopping_obeys_minimum_updates(baseline_config: BaselineConfig) -> None:
    selector = CheckpointSelector(baseline_config.optimization)
    for step in range(200, 2001, 200):
        selector.observe(step=step, patient_macro_dice=0.2, source="validation")
    assert not selector.should_stop(1800)
    selector.observe(step=2200, patient_macro_dice=0.2, source="validation")
    assert selector.should_stop(2200)


def test_test_firewall_requires_lock_and_complete_final_validation(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
) -> None:
    access = RunAccess(locked_protocol, 1, "smoke")
    test_id = access.allowed["test"][0]
    with pytest.raises(RuntimeError, match="sealed"):
        access.read("test", test_id)
    with pytest.raises(ValueError, match="authorized"):
        access.read("train", test_id)
    checkpoint = tmp_path / "local-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic lock fixture; not a model weight file")
    lock = tmp_path / "lock.json"
    write_json(
        lock,
        seal(
            {
                "checkpoint_sha256": file_hash(checkpoint),
                "partition_sha256": locked_protocol.sha256,
                "candidate_index_sha256": locked_protocol.index.sha256,
                "selection_source": "validation",
                "config_sha256": fingerprint(locked_protocol.config.to_dict()),
                "implementation_sha256": implementation_fingerprint(),
                "selected_step": 1,
            }
        ),
    )
    access.lock(lock, checkpoint)
    with pytest.raises(RuntimeError, match="completed validation"):
        access.open_test()
    with pytest.raises(RuntimeError, match="Final validation"):
        access.final_validation_completed()
    for identifier in access.allowed["validation"]:
        access.read("validation", identifier)
    access.final_validation_completed()
    access.open_test()
    access.read("test", test_id)
    with pytest.raises(RuntimeError, match="repeatedly"):
        access.read("test", test_id)


def test_tiny_end_to_end_is_reproducible_and_immutable(
    candidate_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
) -> None:
    config = replace(baseline_config, smoke=SmokeBudget(4, 2, 2, 2))
    protocol = _protocol(candidate_index, config, tmp_path / "partition.json")
    first = run_baseline(
        protocol, k=1, model_seed=config.seeds.smoke, phase="smoke", output_root=tmp_path / "first"
    )
    second = run_baseline(
        protocol, k=1, model_seed=config.seeds.smoke, phase="smoke", output_root=tmp_path / "second"
    )
    a, provenance = read_completed_run(first)
    b, _ = read_completed_run(second)
    assert a["actual_optimization_steps"] == 4
    assert a["selected_state_sha256"] == b["selected_state_sha256"]
    assert (first / "training.jsonl").read_text() == (second / "training.jsonl").read_text()
    assert a["validation"] == b["validation"] and a["test"] == b["test"]
    assert a["patient_training_exposure"] == b["patient_training_exposure"]
    assert a["test_access"]["events"][0]["event"] == "checkpoint_locked"
    assert a["test_access"]["events"][1]["event"] == "final_validation_completed"
    assert a["test_access"]["events"][2]["event"] == "test_evaluation_started"
    for field in (
        "runtime",
        "git_dirty",
        "model",
        "seeds",
        "support_patients",
        "preprocessing_sha256",
        "candidate_index_sha256",
        "optimization",
        "implementation_sha256",
        "patient_assignments",
    ):
        assert field in provenance
    assert provenance["model"]["fraction_trainable"] == 1.0
    assert a["test"]["counts"]["patients"] == 6
    with pytest.raises(FileExistsError):
        run_baseline(
            protocol,
            k=1,
            model_seed=config.seeds.smoke,
            phase="smoke",
            output_root=tmp_path / "first",
        )


@pytest.mark.parametrize("phase", ["calibration", "smoke"])
def test_real_data_engineering_does_not_open_any_test_target(
    private_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
    phase: str,
) -> None:
    payload = read_sealed(private_index.path)
    payload["synthetic"] = False  # Simulate the real-data policy on synthetic arrays only.
    private_index.path.unlink()
    write_json(private_index.path, seal(payload))
    index = CandidateIndex(private_index.path)
    config = replace(
        baseline_config,
        optimization=replace(
            baseline_config.optimization,
            max_steps=2,
            min_steps=1,
            validation_interval=1,
            patience=1,
        ),
        smoke=SmokeBudget(2, 1, 1, 1),
    )
    protocol = _protocol(index, config, tmp_path / "partition.json")
    for identifier in protocol.partition_ids("test"):
        (index.root / index.by_id[identifier].mask_path).write_bytes(b"sealed test target")
        (index.root / index.by_id[identifier].target_qc_path).write_bytes(b"sealed test QC")
    seed = config.seeds.calibration if phase == "calibration" else config.seeds.smoke
    run = run_baseline(protocol, k=1, model_seed=seed, phase=phase, output_root=tmp_path / "runs")
    result, _ = read_completed_run(run)
    assert result["test"] is None
    assert result["test_access"]["read_counts"]["test"] == 0


def test_actual_calibration_freeze_binds_settings_source_and_test_release(
    private_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
) -> None:
    # An adversarial engineering recipe: essentially no weight change creates
    # a stable, non-empty, non-full prediction plateau. This checks the
    # freeze state machine, not whether this recipe is clinically acceptable.
    config = replace(
        baseline_config,
        optimization=replace(
            baseline_config.optimization,
            max_steps=40,
            min_steps=40,
            validation_interval=10,
            patience=1,
            learning_rate=1e-12,
        ),
    )
    protocol = _protocol(private_index, config, tmp_path / "partition.json")
    runs = [
        run_baseline(
            protocol,
            k=k,
            model_seed=config.seeds.calibration,
            phase="calibration",
            output_root=tmp_path / "calibration",
        )
        for k in protocol.calibration_budgets
    ]
    receipt = tmp_path / "frozen.json"
    freeze_recipe(
        protocol, runs, output=receipt, rationale="Synthetic engineering gate verification."
    )
    frozen = verify_frozen_recipe(receipt, protocol)
    assert frozen["test_information_used"] is False
    assert frozen["calibration_budgets"] == [1, 20]
    assert {e["k"] for e in frozen["calibration_evidence"]} == {1, 20}
    changed = replace(config, evaluation=replace(config.evaluation, threshold=0.6))
    changed_protocol = LockedProtocol(private_index, changed, tmp_path / "partition.json")
    with pytest.raises(ValueError, match="no longer matches"):
        verify_frozen_recipe(receipt, changed_protocol)
    release_test(protocol, frozen)
    with pytest.raises(ValueError, match="different frozen recipe"):
        release_test(protocol, {**frozen, "rationale": "changed after test"})
    with pytest.raises(ValueError, match="after definitive test release"):
        freeze_recipe(protocol, runs, output=tmp_path / "illegal.json", rationale="retune")
    with pytest.raises(ValueError, match="Calibration is closed"):
        run_baseline(
            protocol,
            k=1,
            model_seed=config.seeds.calibration,
            phase="calibration",
            output_root=tmp_path / "post-test",
        )


def test_definitive_without_freeze_and_synthetic_definitive_rejected(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen recipe"):
        run_baseline(locked_protocol, k=1, model_seed=17, phase="definitive", output_root=tmp_path)
    with pytest.raises(ValueError, match="Synthetic fixtures"):
        run_baseline(
            locked_protocol,
            k=1,
            model_seed=17,
            phase="definitive",
            output_root=tmp_path,
            frozen_recipe=tmp_path / "nonexistent.json",
        )


def test_completed_results_cannot_be_tampered(
    candidate_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
) -> None:
    config = replace(baseline_config, smoke=SmokeBudget(2, 1, 1, 1))
    protocol = _protocol(candidate_index, config, tmp_path / "partition.json")
    run = run_baseline(
        protocol, k=1, model_seed=config.seeds.smoke, phase="smoke", output_root=tmp_path
    )
    payload = read_sealed(run / "result.json")
    payload["test"]["patient_macro"]["dice"] = 1.0
    (run / "result.json").unlink()
    write_json(run / "result.json", seal(payload))
    assert fingerprint(payload)  # Re-sealing is insufficient; completion anchors the old result.
    with pytest.raises(ValueError, match="Completed result artifact changed"):
        read_completed_run(run)


@pytest.fixture
def definitive_protocol(
    private_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
) -> tuple[LockedProtocol, Path]:
    """Actual tiny calibration, using synthetic arrays under the real-data POLICY.

    The deliberately tiny LR tests state transitions, never real
    medical performance or an acceptable research optimization recipe.
    """
    payload = read_sealed(private_index.path)
    patients = set(private_index.patients[:10])
    payload["records"] = [r for r in payload["records"] if r["patient_id"] in patients]
    payload["synthetic"] = False
    private_index.path.unlink()
    write_json(private_index.path, seal(payload))
    index = CandidateIndex(private_index.path)
    config = replace(
        baseline_config,
        optimization=replace(
            baseline_config.optimization,
            max_steps=12,
            min_steps=8,
            validation_interval=4,
            patience=1,
            learning_rate=1e-12,
        ),
    )
    protocol = _protocol(index, config, tmp_path / "real-policy-partition.json")
    runs = [
        run_baseline(
            protocol,
            k=k,
            model_seed=config.seeds.calibration,
            phase="calibration",
            output_root=tmp_path / "real-policy-calibration",
        )
        for k in protocol.calibration_budgets
    ]
    frozen = tmp_path / "real-policy-frozen.json"
    freeze_recipe(protocol, runs, output=frozen, rationale="Synthetic state-machine test only.")
    return protocol, frozen


def test_definitive_improving_at_cap_cannot_release_or_read_test(
    definitive_protocol: tuple[LockedProtocol, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, frozen = definitive_protocol
    original_evaluate = training.evaluate
    original_load = protocol.index.load
    validation_calls = 0
    loaded: list[str] = []

    def improving_validation(*args: object, **kwargs: object):
        nonlocal validation_calls
        result = original_evaluate(*args, **kwargs)
        if args[2] == "validation":
            validation_calls += 1
            # Adversarial scores around real inference/loading/physical metrics.
            # Final locked validation reproduces the third selected score.
            result["patient_macro"]["dice"] = min(validation_calls, 3) / 10
        return result

    def track_load(identifier: str):
        loaded.append(identifier)
        return original_load(identifier)

    monkeypatch.setattr(training, "evaluate", improving_validation)
    monkeypatch.setattr(protocol.index, "load", track_load)
    with pytest.raises(RuntimeError, match="exhausted.*meaningfully improving"):
        run_baseline(
            protocol,
            k=1,
            model_seed=17,
            phase="definitive",
            output_root=tmp_path / "blocked",
            frozen_recipe=frozen,
        )
    directory = tmp_path / "blocked/definitive/k01/seed017"
    gate = read_sealed(directory / "optimization_gate.json")
    assert gate["status"] == "blocked" and gate["actual_steps"] == 12
    assert gate["diagnostics"]["budget_exhausted_without_patience_plateau"]
    assert gate["test_label_reads"] == 0
    assert set(loaded).isdisjoint(protocol.partition_ids("test"))
    assert (directory / "FAILED.json").is_file()
    assert not (directory / "COMPLETED.json").exists()
    assert not (protocol.index.root / "test_release.json").exists()


def test_adequate_definitive_run_has_locked_once_only_test_and_qc_access(
    definitive_protocol: tuple[LockedProtocol, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, frozen = definitive_protocol
    original_load = protocol.index.load
    test_loads: list[str] = []

    def track_load(identifier: str):
        if identifier in protocol.partition_ids("test"):
            test_loads.append(identifier)
        return original_load(identifier)  # Verifies full QC after RunAccess authorization.

    monkeypatch.setattr(protocol.index, "load", track_load)
    run = run_baseline(
        protocol,
        k=1,
        model_seed=17,
        phase="definitive",
        output_root=tmp_path / "ready",
        frozen_recipe=frozen,
    )
    result, provenance = read_completed_run(run)
    gate = read_sealed(run / "optimization_gate.json")
    assert gate["status"] == "passed" and gate["test_label_reads"] == 0
    assert len(test_loads) == len(set(test_loads)) == len(protocol.partition_ids("test"))
    events = [e["event"] for e in result["test_access"]["events"]]
    assert events == [
        "checkpoint_locked",
        "final_validation_completed",
        "optimization_adequacy_passed",
        "test_evaluation_started",
        "test_evaluation_completed",
    ]
    assert result["test"] is not None and provenance["model"]["fraction_trainable"] == 1


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_numerical_failure_never_exposes_test(
    definitive_protocol: tuple[LockedProtocol, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    protocol, frozen = definitive_protocol
    original_model = training.build_random_model
    loaded: list[str] = []
    original_load = protocol.index.load

    def track_load(identifier: str):
        loaded.append(identifier)
        return original_load(identifier)

    if failure == "gradient":

        def bad_gradient(*args: object, **kwargs: object):
            model, audit = original_model(*args, **kwargs)
            next(model.parameters()).register_hook(lambda gradient: gradient * float("nan"))
            return model, audit

        monkeypatch.setattr(training, "build_random_model", bad_gradient)
    else:
        original_loss = training.DiceCELoss

        class BadLoss(original_loss):
            def forward(self, *args: object, **kwargs: object):
                return super().forward(*args, **kwargs) * float("nan")

        monkeypatch.setattr(training, "DiceCELoss", BadLoss)
    monkeypatch.setattr(protocol.index, "load", track_load)
    with pytest.raises(RuntimeError, match="[Nn]on.finite"):
        run_baseline(
            protocol,
            k=1,
            model_seed=17,
            phase="definitive",
            output_root=tmp_path / "nonfinite",
            frozen_recipe=frozen,
        )
    assert set(loaded).isdisjoint(protocol.partition_ids("test"))
    assert not (protocol.index.root / "test_release.json").exists()
    assert (tmp_path / "nonfinite/definitive/k01/seed017/FAILED.json").exists()


@pytest.mark.parametrize("drift", ["recipe", "source", "index", "partition"])
def test_drift_after_training_still_blocks_pretest_release(
    definitive_protocol: tuple[LockedProtocol, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    protocol, frozen = definitive_protocol
    original_evaluate = training.evaluate

    def drift_during_final_validation(*args: object, **kwargs: object):
        result = original_evaluate(*args, **kwargs)
        if args[1].state == "locked":
            if drift == "source":
                monkeypatch.setattr(training, "implementation_fingerprint", lambda: "changed")
                monkeypatch.setattr(
                    "medfm_adapt3d.baseline.selection.implementation_fingerprint", lambda: "changed"
                )
            else:
                path = {"recipe": frozen, "index": protocol.index.path, "partition": protocol.path}[
                    drift
                ]
                payload = read_sealed(path)
                if drift == "recipe":
                    payload["rationale"] = "changed after checkpoint selection"
                elif drift == "index":
                    payload["source_audit"]["evidence_source"] = "changed"
                else:
                    payload["support_seed"] += 1
                path.unlink()
                write_json(path, seal(payload))
        return result

    monkeypatch.setattr(training, "evaluate", drift_during_final_validation)
    with pytest.raises(ValueError, match="changed|drift|no longer matches"):
        run_baseline(
            protocol,
            k=1,
            model_seed=17,
            phase="definitive",
            output_root=tmp_path / "drift",
            frozen_recipe=frozen,
        )
    assert not (protocol.index.root / "test_release.json").exists()
    gate = read_sealed(tmp_path / "drift/definitive/k01/seed017/optimization_gate.json")
    assert gate["test_label_reads"] == 0


def test_freeze_cannot_mix_different_recipes_for_small_and_large_k(
    definitive_protocol: tuple[LockedProtocol, Path],
    tmp_path: Path,
) -> None:
    protocol, _ = definitive_protocol
    alternative = replace(
        protocol.config,
        optimization=replace(protocol.config.optimization, learning_rate=2e-12),
    )
    other_protocol = LockedProtocol(protocol.index, alternative, protocol.path)
    largest = run_baseline(
        other_protocol,
        k=5,
        model_seed=alternative.seeds.calibration,
        phase="calibration",
        output_root=tmp_path / "alternative-recipe",
    )
    smallest = tmp_path / "real-policy-calibration/calibration/k01/seed997"
    with pytest.raises(ValueError, match="ONE current recipe"):
        freeze_recipe(
            protocol,
            [smallest, largest],
            output=tmp_path / "mixed.json",
            rationale="Attempt to mix K-specific recipes must fail.",
        )
    assert not (tmp_path / "mixed.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint_selection_complete", False),
        ("final_validation_complete", False),
        ("minimum_steps_satisfied", False),
        ("numerically_finite", False),
        ("budget_exhausted_without_patience_plateau", True),
        ("all_validation_predictions_empty", True),
        ("all_validation_predictions_full", True),
        ("zero_validation_overlap", True),
        ("unvisited_training_patients", 1),
        ("unvisited_training_targets", 1),
        ("patient_exposure_balanced", False),
    ],
)
def test_every_prespecified_adequacy_failure_blocks_release(key: str, value: object) -> None:
    good = {
        "checkpoint_selection_complete": True,
        "final_validation_complete": True,
        "minimum_steps_satisfied": True,
        "numerically_finite": True,
        "budget_exhausted_without_patience_plateau": False,
        "all_validation_predictions_empty": False,
        "all_validation_predictions_full": False,
        "zero_validation_overlap": False,
        "unvisited_training_patients": 0,
        "unvisited_training_targets": 0,
        "patient_exposure_balanced": True,
    }
    require_optimization_adequacy(good)
    with pytest.raises(RuntimeError):
        require_optimization_adequacy({**good, key: value})
