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

Raw medical images, generated manifests, caches, model weights, and experiment
outputs are intentionally excluded from Git. Tests and validation scripts use
synthetic data and make no claim of clinical performance.

## Development Checks

```bash
python -m pytest
ruff check .
python scripts/validate_research_contract.py
python scripts/validate_preprocessing_contract.py
```

## Licence and Use Restrictions

Copyright (c) 2026 Chandima Liyana. All rights reserved.

This repository is source-available for limited research evaluation; it is not
open-source software and is not licensed for clinical use, redistribution,
derivative works, or incorporation into another project without prior written
permission. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
