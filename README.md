# MedFM-Adapt3D

**Trustworthy, label-efficient adaptation of 3D medical models for pulmonary-nodule analysis under annotation scarcity and distribution shift.**

MedFM-Adapt3D is a PyTorch/MONAI research codebase.

The first milestone establishes the experimental safeguards that later
foundation-model adaptation experiments must obey:

- patient-level leakage prevention,
- reproducible nested few-shot cohorts,
- dataset-lineage checks,
- trainable-parameter accounting,
- reproducibility and runtime provenance.

## Research direction

- **Primary data:** LIDC-IDRI thoracic CT
- **Detection protocol:** LUNA16
- **External generalisation:** LNDb
- **Later extension:** LUNA25

LUNA16 is derived from LIDC-IDRI and is therefore not treated as an independent
external dataset.

## Commit-1 principle

Before optimizing Dice, FROC, calibration, or uncertainty, the project makes
invalid comparisons difficult to run.

See [`docs/research_protocol.md`](docs/research_protocol.md) for the scientific
contract.

## Development checks

```bash
python scripts/validate_research_contract.py
pytest
ruff check .