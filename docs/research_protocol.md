# MedFM-Adapt3D Research Protocol — Commit 1

## Scope

MedFM-Adapt3D studies trustworthy, label-efficient adaptation of 3D medical
models for pulmonary-nodule analysis.

The initial research domain is thoracic CT using LIDC-IDRI, with the LUNA16
protocol used for standardized detection benchmarking.

LNDb is reserved for later independent external generalisation, and LUNA25 is
a later screening/risk extension.

## Commit-1 scientific contract

Before training a model, the repository enforces four invariants:

1. **Patient-level separation.**
   A patient must never occur in more than one of the adaptation, validation,
   or test partitions.

2. **Nested K-shot cohorts.**
   K-shot budgets are defined in patients and constructed from one deterministic
   ordering, so a smaller support set is contained in every larger support set.

3. **Dataset-lineage honesty.**
   LUNA16 is derived from LIDC-IDRI and therefore must not be presented as an
   independent external-validation cohort for an LIDC-developed model.

4. **Adaptation-budget transparency.**
   Every adaptation strategy must report the number and percentage of trainable
   parameters.

## Why no medical images are committed

Public medical-imaging datasets remain governed by their source licences and
access terms.

Raw DICOM/NIfTI data, model checkpoints, caches, and experiment outputs are
intentionally excluded from Git.

The repository will instead version manifests, data-selection logic,
preprocessing configuration, and provenance needed to reconstruct an experiment.

## What Commit 1 does not claim

This commit contains no clinical-performance result, no foundation-model
comparison, and no claim of state-of-the-art performance.

The synthetic identifiers used by the smoke test exist only to verify
experimental plumbing.