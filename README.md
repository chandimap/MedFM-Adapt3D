# MedFM-Adapt3D

**Trustworthy, Label-Efficient Adaptation of 3D Medical Models for Pulmonary-Nodule Analysis under Annotation Scarcity and Distribution Shift.**

MedFM-Adapt3D is a Python 3.12, PyTorch, and MONAI research codebase for
auditable 3D thoracic-CT experiments.

## Implemented Research Safeguards

- patient-level leakage prevention,
- reproducible nested few-shot cohorts,
- dataset-lineage checks,
- trainable-parameter accounting,
- reproducibility and runtime provenance,
- exact LIDC-IDRI DICOM/XML identity reconciliation,
- reader-specific annotation preservation,
- SOP-linked reader-contour rasterization without consensus collapse,
- explicit DICOM stored-value-to-HU conversion,
- physical-coordinate image/mask alignment checks,
- LPS orientation and 1 mm isotropic spacing normalization,
- linear CT and nearest-neighbour mask interpolation,
- physical candidate-centred 3D crops, and
- size-aware tiny-nodule volume, centroid, and crop-preservation gates.

LUNA16 is derived from LIDC-IDRI and is not represented as an independent
external dataset.

## Preprocessing Contract

The version-controlled preprocessing protocol is
[`configs/preprocessing/lidc_candidate.yaml`](configs/preprocessing/lidc_candidate.yaml).
It fixes coordinate convention, spacing, HU window, interpolation semantics,
crop extent, and mask-preservation thresholds before model experiments.

## Conventional Random Reference

The baseline implements **oracle-centred localized 3D pulmonary-nodule
segmentation** with randomly initialized MONAI SegResNet. Reference annotations
supply candidate centres. Nested budgets count 1, 5, 10, or 20 labelled patients;
five prespecified initialization seeds share locked partitions and support sets.
Patient-balanced reader rotation, step-based optimization, validation-only
checkpoint selection, pre-test adequacy gates, physical metrics, and immutable local artifacts provide
inspectable reference-model evidence.

Real-data engineering uses train/validation only. Definitive testing requires
a common frozen recipe calibrated at the smallest/largest feasible K. A run
still improving at its optimizer-step cap cannot release test. Input metadata
contains CT geometry and the declared oracle centre; additional mask QC remains
restricted to authorized target loads. Spatial reflections are disabled.

Repeated model seeds quantify initialization/optimization variability conditional
on one fixed patient partition and one fixed nested support ordering. They do
not quantify alternative partitions, alternative K-patient support samples, or
patient/population uncertainty. All seeds remain inspectable. The small pilot can
verify software without satisfying every label budget.

See the [baseline protocol](docs/random_baseline_protocol.md),
[Windows local release gate](docs/random_baseline_local_release.md), and
[verification evidence](docs/random_baseline_verification.md).

This establishes a conventional reference, not whole-scan detection, screening,
malignancy diagnosis, clinical deployability, or evidence that any K is clinically
sufficient. A later comparison with a different foundation-model architecture
cannot attribute performance differences solely to pretraining. No real LIDC
baseline performance has been measured in the supplied verification environment.

Raw medical images, generated manifests, caches, model weights, and experiment
outputs are intentionally excluded from Git. Tests and validation scripts use
synthetic data and make no claim of clinical performance.

## Development Checks

```bash
python -m pytest
ruff check .
python -m compileall -q src scripts tests
python scripts/validate_research_contract.py
python scripts/validate_preprocessing_contract.py
python scripts/validate_random_baseline_contract.py
python scripts/smoke_random_baseline.py
```

## Licence and Use Restrictions

Copyright (c) 2026 Chandima Liyana. All rights reserved.

This repository is source-available for limited research evaluation; it is not
open-source software and is not licensed for clinical use, redistribution,
derivative works, or incorporation into another project without prior written
permission. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
