# Frozen Representation Verification

Verification date: 25 October 2026. Inspected repository base:
`7617657605634d4e0e9fbfeb0f90eaa7fbc2a66c`.

The exact package-source fingerprint, environment, regression count and aggregate
official-weight smoke findings are recorded in
[`verification.json`](../evidence/frozen_representation/verification.json).
These describe the supplied Linux CPU verification, not checks personally run by
another researcher or checks on Windows/CUDA.

## Observed official-checkpoint integration

The released `supervised_suprem_segresnet_2100.pth` was downloaded from the pinned
upstream revision. Its 56,500,623 bytes matched the published LFS SHA-256. All
58 expected encoder tensors loaded with exact shapes/dtypes and finite values:
4,399,344 frozen encoder parameters. Source decoder tensors were explicitly
discarded; only the new 241-parameter head was trainable.

The paired end-to-end smoke used ten synthetic patients, K=1, head seed 991,
the existing patient split/support protocol, and the configured 12-update smoke
budget. Both heads changed and both encoders remained bitwise identical under
their state fingerprints. Head initialization matched across arms. Neither arm
requested held-out image, mask or target-QC arrays.

**The pretrained smoke's selected head had zero validation overlap. Its
optimization-adequacy gate was blocked.** The random-control smoke's gate passed.
Successful software execution is therefore reported separately from adequate
optimization. The comparison records both statuses and withholds the Dice
difference. These short synthetic runs establish neither useful transfer nor a
ranking on clinical data. The blocked outcome is retained, not discarded or
relabelled as successful performance.

## Regression and Adversarial Verification

The full suite includes the existing cohort, preprocessing and reference-model
tests plus representation-specific cases for:

- exact complete checkpoint loading and safe rejection before deserialization;
- missing/unknown keys, incompatible shapes/dtypes, non-finite tensors and mixed prefixes;
- exact agreement with MONAI's four-scale encoder feature semantics;
- frozen parameters, gradients and mode, with detected state violations;
- deterministic and separately seeded representation/head construction;
- odd-sized crop-grid preservation and invalid/non-finite input rejection;
- source-record/claim restrictions and no silent random initialization fallback;
- repeated head-adaptation reproducibility, output immutability and receipt drift;
- unsupported phases/seeds/arms/budgets rejected before output creation;
- zero test loads, including corrupt held-out files under a simulated real-data policy;
- non-finite training failure receipts and source-drift rejection; and
- every passed/blocked pair of optimization gates before reporting a difference.

Synthetic arrays exercising a real-data access policy remain synthetic evidence.
The suite does not establish clinical chronology, performance, population
generalization, model calibration, bias, robustness or clinical readiness.

The research, preprocessing, random-reference and frozen-representation contract
scripts passed, together with lint, compilation and dependency checks. The
changed-files package and source distribution are checked for required source,
configuration, tests and aggregate evidence. Private teaching materials and
pretrained/adapted encoder weights are excluded from the public package.

## Reproduction

From the repository root in the project's Python 3.12 environment:

```bash
python -m pytest
python -m ruff check .
python -m compileall -q src scripts tests
python scripts/validate_research_contract.py
python scripts/validate_preprocessing_contract.py
python scripts/validate_random_baseline_contract.py
python scripts/validate_frozen_representation_contract.py
python -m medfm_adapt3d.foundation.cli audit
```

The following explicitly requires the pinned official checkpoint, downloaded
locally under its recorded upstream terms:

```bash
python scripts/smoke_frozen_representation.py --checkpoint checkpoints/supervised_suprem_segresnet_2100.pth --output outputs/frozen_representation_synthetic
```

Choose a fresh output directory for a new run. Execution outputs contain complete
local receipts and should remain private. A rerun on another hardware/software
stack should be assessed through the recorded settings, invariants and scope;
binary artifact identity across stacks is not promised.

Patient-level pretraining overlap and exact source-input compatibility remain
**NOT_ESTABLISHED**. No real LIDC-IDRI experiment or clean-generalization claim is
reported by this verification. The clinical task remains localized segmentation
with reference-derived centres, not autonomous detection or clinical deployment.
