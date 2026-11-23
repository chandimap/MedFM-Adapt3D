"""Bounded training-only memorization of a single locked LIDC reader target on CPU."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

from medfm_adapt3d.baseline.artifacts import file_hash, implementation_fingerprint, seal, write_json
from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import load_baseline_config
from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.baseline.selection import RunAccess
from medfm_adapt3d.engineering.reproducibility import collect_runtime_provenance
from medfm_adapt3d.foundation.feature_adapter import (
    FeatureRecipe,
    configure_feature_arm,
    trainable_groups,
)
from medfm_adapt3d.foundation.feature_audit import CpuMemory
from medfm_adapt3d.foundation.model import build_frozen_model
from medfm_adapt3d.foundation.provenance import load_model_record

READOUTS = (0, 12, 64, 128, 256, 512)


def run(index: Path, partitions: Path, checkpoint: Path, arm: str) -> dict:
    torch.set_num_threads(2)
    cfg = load_baseline_config(Path("configs/baseline/random_segresnet.yaml"))
    protocol = LockedProtocol(CandidateIndex(index), cfg, partitions)
    barrier = protocol.index.root / "test_release.json"
    if protocol.index.payload["synthetic"] or barrier.exists():
        raise ValueError("This diagnostic requires a real, test-closed private cohort.")
    access = RunAccess(protocol, 1, "calibration")
    candidate = min(protocol.support_ids(1))
    image, mask = access.read("train", candidate)
    x, y = image.unsqueeze(0), mask.unsqueeze(0).float()
    record = load_model_record(Path("configs/foundation/suprem_segresnet.yaml"))
    recipe = FeatureRecipe.load(Path("configs/foundation/feature_adapter.yaml"))
    budget = cfg.budget("smoke")
    start = time.perf_counter()
    with CpuMemory() as memory:
        model, init = build_frozen_model(
            record, arm="suprem_frozen", head_seed=cfg.seeds.smoke, checkpoint=checkpoint
        )
        configure_feature_arm(model, init, arm, recipe)
        model.train()
        shape = x.shape[-3:]
        padding = tuple(v for n in reversed(shape) for v in (0, -n % 8))
        x_pad = F.pad(x, padding, mode="replicate")
        with torch.no_grad():
            features = model.encoder(x_pad)

        def forward():
            logits = model.head(features, tuple(x_pad.shape[-3:]))
            return logits[..., : shape[0], : shape[1], : shape[2]]

        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay
        )
        loss_fn = DiceCELoss(
            sigmoid=True,
            lambda_dice=budget.dice_weight,
            lambda_ce=budget.ce_weight,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        readings = []
        for step in range(READOUTS[-1] + 1):
            logits = forward()
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite single-target objective.")
            if step in READOUTS:
                with torch.no_grad():
                    p = logits.sigmoid()
                    pred, truth = p >= cfg.evaluation.threshold, y.bool()
                    overlap = int((pred & truth).sum())
                    readings.append(
                        {
                            "step": step,
                            "loss": float(loss.detach()),
                            "reference_voxels": int(truth.sum()),
                            "predicted_voxels": int(pred.sum()),
                            "overlap_voxels": overlap,
                            "dice": 2 * overlap / int(pred.sum() + truth.sum()),
                            "mean_p_foreground": float(p[truth].mean()),
                            "mean_p_background": float(p[~truth].mean()),
                        }
                    )
                    if step in (0, READOUTS[-1]):
                        torch.testing.assert_close(logits, model(x), rtol=0, atol=0)
            if step == READOUTS[-1]:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head.parameters(), budget.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
        model.assert_frozen(init["encoder_initial_sha256"])
        if barrier.exists() or access.read_counts != {"train": 1, "validation": 0, "test": 0}:
            raise RuntimeError("Single-target diagnostic access scope changed.")
    return seal(
        {
            "study": "fixed single training mask memorization",
            "synthetic": False,
            "arm": arm,
            "candidate_hash": candidate,
            "partition_sha256": protocol.sha256,
            "candidate_index_sha256": protocol.index.sha256,
            "selection": "first lexicographically ordered K=1 reader target",
            "threshold": cfg.evaluation.threshold,
            "terminal_criterion": 0.80,
            "terminal_criterion_met": readings[-1]["dice"] >= 0.80,
            "head_seed": cfg.seeds.smoke,
            "effective_optimization": asdict(budget)
            | {
                "max_steps": 512,
                "selection": "none; fixed readouts",
                "min_steps": 512,
                "validation_interval": None,
                "patience": None,
            },
            "feature_recipe": asdict(recipe),
            "checkpoint_sha256": record.checkpoint_sha256,
            "encoder_before_sha256": init["encoder_initial_sha256"],
            "encoder_after_sha256": state_fingerprint(model.encoder),
            "groups_changed": {
                g: state_fingerprint(m) != init["trainable_initial_sha256"][g]
                for g, m in trainable_groups(model).items()
            },
            "uncached_logits_bitwise_equal_at_steps": [0, 512],
            "frozen_features_cached": True,
            "read_counts": access.read_counts,
            "training_presentations": 512,
            "readouts": readings,
            "environment": asdict(collect_runtime_provenance()),
            "implementation_sha256": implementation_fingerprint(),
            "script_sha256": file_hash(Path(__file__)),
            "wall_seconds": time.perf_counter() - start,
            "cpu_memory": memory.report(),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("index", "partitions", "checkpoint", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--arm", choices=["suprem_frozen", "suprem_feature"], required=True)
    args = parser.parse_args()
    write_json(args.output, run(args.index, args.partitions, args.checkpoint, args.arm))
