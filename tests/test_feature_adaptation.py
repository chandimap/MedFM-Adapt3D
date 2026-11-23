"""Adversarial tests of the pointwise feature experiment and its interpretation."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from medfm_adapt3d.baseline.artifacts import read_sealed
from medfm_adapt3d.baseline.config import SmokeBudget
from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.foundation import feature_experiment, training
from medfm_adapt3d.foundation.feature_adapter import (
    FEATURE_ARMS,
    FeatureRecipe,
    ResidualFeatureProbe,
    configure_feature_arm,
    trainable_groups,
)
from medfm_adapt3d.foundation.feature_experiment import paired_difference, restore_feature_model
from medfm_adapt3d.foundation.model import build_frozen_model
from medfm_adapt3d.foundation.provenance import load_model_record
from medfm_adapt3d.foundation.training import read_probe_run, run_probe

RECIPE = FeatureRecipe.load(Path("configs/foundation/feature_adapter.yaml"))
RECORD = load_model_record(Path("configs/foundation/suprem_segresnet.yaml"))


@pytest.mark.parametrize(
    "change",
    [
        {"method": "unknown"},
        {"feature_level": 1},
        {"feature_level": False},
        {"bottleneck": 16},
        {"bottleneck": 0},
        {"bottleneck": True},
        {"adapter_seed": -1},
        {"adapter_seed": 2**32},
    ],
)
def test_unsupported_recipe(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(RECIPE, **change)


@pytest.mark.parametrize("yaml_text", ["[]", "method: invalid", "unexpected: true\nmethod: bad"])
def test_recipe_schema_is_closed(tmp_path: Path, yaml_text: str) -> None:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError):
        FeatureRecipe.load(path)


def test_identity_shapes_rng_and_nonzero_gradient_path() -> None:
    model, init = build_frozen_model(RECORD, arm="random_frozen", head_seed=991)
    image = torch.linspace(-1, 1, 17 * 18 * 19).reshape(1, 1, 17, 18, 19)
    expected = model(image).detach()
    old_rng = torch.get_rng_state().clone()
    configure_feature_arm(model, init, "random_feature", RECIPE)
    assert torch.equal(old_rng, torch.get_rng_state())
    adapter = model.head
    assert isinstance(adapter, ResidualFeatureProbe)
    assert torch.count_nonzero(adapter.up.weight) == 0
    assert torch.count_nonzero(adapter.down.weight) > 0
    assert state_fingerprint(adapter.probe) == init["head_initial_sha256"]
    torch.testing.assert_close(model(image), expected, atol=0, rtol=0)
    budget = init["parameter_budget"]
    assert budget["total_parameters"] == 4399344 + 241 + 128
    assert budget["frozen_parameters"] == budget["encoder_parameters"] == 4399344
    assert budget["adapter_parameters"] == budget["added_parameters"] == 128
    assert budget["trainable_parameters"] == 369
    assert budget["trainable_percentage"] == pytest.approx(100 * 369 / budget["total_parameters"])
    # Zero decay separates a data gradient from an AdamW weight-decay update.
    opt = torch.optim.SGD(model.head.parameters(), lr=0.01)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        logits = model(image)
        logits.square().mean().backward()
        assert adapter.up.weight.grad.norm() > 0
        assert (adapter.down.weight.grad.norm() > 0).item() == (step > 0)
        assert adapter.probe.projections[0].weight.grad.norm() > 0
        opt.step()
    assert all(
        state_fingerprint(m) != init["trainable_initial_sha256"][n]
        for n, m in trainable_groups(model).items()
    )
    model.assert_frozen(init["encoder_initial_sha256"])
    with pytest.raises(ValueError, match="four"):
        adapter((), (16, 16, 16))
    with pytest.raises(ValueError, match="16 channels"):
        adapter((torch.ones(1, 2, 1, 1, 1),) * 4, (16, 16, 16))


def test_representation_and_task_origins_cannot_be_swapped() -> None:
    model, init = build_frozen_model(RECORD, arm="random_frozen", head_seed=991)
    with pytest.raises(ValueError, match="conflicts"):
        configure_feature_arm(model, init, "suprem_feature", RECIPE)
    next(model.encoder.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="Trainable set"):
        configure_feature_arm(model, init, "random_feature", RECIPE)


def _short(protocol: LockedProtocol) -> LockedProtocol:
    return LockedProtocol(
        protocol.index, replace(protocol.config, smoke=SmokeBudget(3, 2, 2, 2)), protocol.path
    )


def _run(protocol: LockedProtocol, output: Path) -> Path:
    return run_probe(
        protocol,
        RECORD,
        arm="random_feature",
        k=1,
        head_seed=991,
        phase="smoke",
        output_root=output,
        feature_recipe=RECIPE,
    )


def test_real_path_isolated_updates_selected_reload_and_evidence(
    locked_protocol: LockedProtocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _short(locked_protocol)
    original = protocol.index.load
    forbidden = set(protocol.partition_ids("test"))
    accessed = []

    def guard(identifier: str):
        assert identifier not in forbidden
        accessed.append(identifier)
        return original(identifier)

    monkeypatch.setattr(protocol.index, "load", guard)
    run = _run(protocol, tmp_path / "runs")
    result, p = read_probe_run(run)
    audit = result["feature_audit"]
    assert set(accessed) == set(protocol.support_ids(1)) | set(protocol.partition_ids("validation"))
    assert result["test"] is None and result["test_access"]["read_counts"]["test"] == 0
    assert result["encoder_final_sha256"] == p["model"]["encoder_initial_sha256"]
    assert audit["adaptation_status"] == "PASS"
    assert all(audit["selected_group_changed"].values())
    assert set(audit["nonzero_gradient_groups"]) == {"down", "up", "probe"}
    assert len(audit["first_training_feature_statistics"]) == 4
    assert audit["validation_trajectory"][-1]["selected_replay"]
    assert audit["training_steps_seconds"] > 0 and audit["validation_seconds"] > 0
    memory = result["cpu_memory"]
    assert (
        memory["peak_observed_rss_bytes"]
        >= max(memory["rss_start_bytes"], memory["rss_end_bytes"])
        > 0
    )
    assert memory["samples"] >= 2 and memory["gpu_memory_bytes"] is None
    restored = restore_feature_model(run, RECORD, checkpoint=None)
    expected = sorted(n for n, p in restored.named_parameters() if p.requires_grad)
    assert result["optimizer_parameters"] == expected
    selected = torch.load(run / "selected_task.pt", weights_only=True, map_location="cpu")
    assert set(selected) == set(restored.head.state_dict())
    assert all(not k.startswith("encoder.") for k in selected)
    assert (
        read_sealed(run / "checkpoint_lock.json")["selected_logits_sha256"]
        == audit["reloaded_logits_sha256"]
    )
    again = _run(protocol, tmp_path / "repeat")
    assert (
        read_probe_run(again)[0]["feature_audit"]["selected_group_sha256"]
        == audit["selected_group_sha256"]
    )
    with pytest.raises(FileExistsError):
        _run(protocol, tmp_path / "runs")
    with (run / "selected_task.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        restore_feature_model(run, RECORD, checkpoint=None)


def test_first_step_selection_does_not_attest_down_projection_adaptation(
    locked_protocol: LockedProtocol, tmp_path: Path
) -> None:
    cfg = replace(locked_protocol.config, smoke=SmokeBudget(1, 1, 1, 1))
    protocol = LockedProtocol(locked_protocol.index, cfg, locked_protocol.path)
    result, _ = read_probe_run(_run(protocol, tmp_path))
    audit = result["feature_audit"]
    assert audit["adaptation_status"] == "FAIL"
    assert "down" not in audit["nonzero_gradient_groups"]
    assert audit["selected_group_changed"]["up"]


def test_nonfinite_feature_loss_aborts(
    locked_protocol: LockedProtocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Nonfinite(training.DiceCELoss):
        def forward(self, *args: object, **kwargs: object):
            return super().forward(*args, **kwargs) * float("nan")

    monkeypatch.setattr(training, "DiceCELoss", Nonfinite)
    with pytest.raises(RuntimeError, match="Non-finite"):
        _run(_short(locked_protocol), tmp_path)
    assert len(list(tmp_path.rglob("FAILED.json"))) == 1
    assert not list(tmp_path.rglob("COMPLETED.json"))


@pytest.mark.parametrize(
    "blocked,changed,expected", [(True, True, None), (False, False, None), (False, True, -0.25)]
)
def test_blocked_collapse_and_verified_negative_result(
    blocked: bool, changed: bool, expected: float | None
) -> None:
    control = {
        "adequacy": "PASSED",
        "feature_audit": {"adaptation_status": "PASS"},
        "validation_patient_macro": {"dice": 0.75},
    }
    intervention = {
        "adequacy": "BLOCKED" if blocked else "PASSED",
        "feature_audit": {"adaptation_status": "PASS" if changed else "FAIL"},
        "validation_patient_macro": {"dice": 0.50},
    }
    assert paired_difference(control, intervention) == expected


@pytest.mark.parametrize("fault", ["cohort", "logits", "capacity", "absent"])
def test_unmatched_pairs_cannot_be_summarized(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Only comparison bookkeeping is mocked. No pretrained performance is inferred.
    def arms(path: Path, arm: str):
        model = {
            "encoder_specification": {},
            "head_initial_sha256": "fixed",
            "feature_recipe": asdict(RECIPE),
            "input_policy": "same",
        }
        p = {k: None for k in feature_experiment.MATCH_FIELDS}
        p.update(
            model=model,
            partition_sha256="new" if fault == "cohort" and arm == "random_feature" else "same",
        )
        adapted = arm.endswith("_feature")
        a = {
            "encoder_initial_sha256": arm.split("_")[0],
            "parameter_budget": {
                "trainable": 370
                if fault == "capacity" and arm == "random_feature"
                else 369
                if adapted
                else 241
            },
            "feature_audit": {
                "initial_logits_sha256": "different"
                if fault == "logits" and arm == "suprem_feature"
                else "match",
                "initial_group_sha256": {},
                "adaptation_status": "PASS",
            },
            "adequacy": "BLOCKED",
        }
        return a, p

    monkeypatch.setattr(feature_experiment, "_arm_summary", arms)
    paths = {arm: Path(arm) for arm in FEATURE_ARMS}
    if fault == "absent":
        del paths["suprem_feature"]
    with pytest.raises(ValueError):
        feature_experiment.summarize(paths)


def test_later_gradients_cannot_validate_decay_only_at_selected_checkpoint(
    locked_protocol: LockedProtocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a validation tie at every step and make AdamW decay clearly measurable.
    evaluate_actual = training.evaluate

    def validation_tie(*args, **kwargs):
        result = evaluate_actual(*args, **kwargs)
        result["patient_macro"]["dice"] = 0.0
        return result

    monkeypatch.setattr(training, "evaluate", validation_tie)
    cfg = replace(
        locked_protocol.config,
        optimization=replace(locked_protocol.config.optimization, weight_decay=0.05),
        smoke=SmokeBudget(3, 3, 1, 3),
    )
    proto = LockedProtocol(locked_protocol.index, cfg, locked_protocol.path)
    path = _run(proto, tmp_path)
    result, _ = read_probe_run(path)
    a = result["feature_audit"]
    assert result["selected_step"] == 1 and result["actual_optimization_steps"] == 3
    assert a["selected_group_changed"]["down"] and "down" in a["selected_nonzero_update_groups"]
    assert (
        "down" in a["nonzero_gradient_groups"]
        and "down" not in a["selected_nonzero_gradient_groups"]
    )
    assert a["adaptation_status"] == "FAIL"
    assert feature_experiment._arm_summary(path, "random_feature")[0]["feature_audit"] == a


@pytest.mark.parametrize("fault", ["checkpoint_provenance", "gate_status", "selected_gradients"])
def test_resealed_inconsistent_worker_receipts_are_rejected(
    fault: str, locked_protocol: LockedProtocol, tmp_path: Path
) -> None:
    from medfm_adapt3d.baseline.artifacts import file_hash, read_json, seal, write_json

    path = _run(_short(locked_protocol), tmp_path)
    filename = {
        "checkpoint_provenance": "checkpoint_lock.json",
        "gate_status": "optimization_gate.json",
        "selected_gradients": "result.json",
    }[fault]
    target = path / filename
    receipt = read_sealed(target)
    if fault == "checkpoint_provenance":
        receipt["partition_sha256"] = "0" * 64
    elif fault == "gate_status":
        receipt["status"] = "passed" if receipt["status"] == "blocked" else "blocked"
    else:
        receipt["feature_audit"]["selected_nonzero_gradient_groups"] = []
    target.unlink()
    write_json(target, seal(receipt))
    completion = path / "COMPLETED.json"
    links = read_json(completion)
    links[filename] = file_hash(target)
    completion.unlink()
    write_json(completion, links)
    with pytest.raises(ValueError, match="provenance|optimization gate|gradient/update"):
        feature_experiment._arm_summary(path, "random_feature")


@pytest.mark.parametrize("fault", ["exposure", "blocked_difference", "primary_threshold"])
def test_public_evidence_rejects_contradictory_scientific_claims(fault: str) -> None:
    import runpy

    validator = runpy.run_path("scripts/validate_feature_evidence.py")["validate_comparison"]
    c = read_sealed(Path("evidence/feature_adaptation/followup/ct_k1.json"))
    if fault == "exposure":
        c["arms"]["random_frozen"]["feature_audit"]["draw_history"][0]["ordered_target_sha256"] = (
            "a" * 64
        )
    elif fault == "blocked_difference":
        c["selected_validation_dice_difference_feature_minus_frozen"]["suprem"] = 0.98
    else:
        c["baseline_configuration"]["evaluation"]["threshold"] = 0.1
    with pytest.raises(ValueError):
        validator(c, extended=True)
