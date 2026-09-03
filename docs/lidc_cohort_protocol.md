# LIDC-IDRI Cohort Construction Protocol

## Scope

The LIDC-IDRI cohort is represented as an auditable hierarchy of patient,
DICOM study, CT series, DICOM instances, reading sessions and reader-specific
pulmonary-nodule annotations.

The cohort builder does not infer model targets or collapse reader disagreement.
Its purpose is to establish traceable experimental samples before model fitting.

## Series identity

CT series are linked to LIDC annotations using exact DICOM
`StudyInstanceUID` and `SeriesInstanceUID` values.

Folder names alone are not treated as sufficient series identity.

Each selected series must have:

- one patient identifier,
- one study UID,
- one series UID,
- unique SOP Instance UIDs,
- consistent image dimensions,
- valid positive pixel spacing,
- valid positive slice thickness,
- multiple spatially distinct image positions.

## Annotation identity

LIDC reading sessions are preserved separately.

Reader order is treated as scan-local and is not interpreted as a persistent
radiologist identity across patients.

For each `unblindedReadNodule`, the manifest preserves:

- reader-local annotation identifier,
- referenced SOP Instance UIDs,
- ROI z positions,
- inclusion/exclusion status,
- contour coordinates,
- available nodule characteristics.

Reader-specific characteristics are observations, not pathology-confirmed
ground-truth labels.

## Annotation-to-image QC

Every SOP Instance UID referenced by an annotation must occur in the linked CT
series. Missing image references produce a failed QC state rather than an
implicit reassignment to another CT reconstruction.

## LUNA16 relationship

LUNA16 is derived from LIDC-IDRI and is not considered an independent external
dataset.

The LUNA16 maximum slice-thickness criterion of 2.5 mm is recorded as an audit
attribute. It is not used to redefine the complete LIDC-IDRI cohort.

## Experimental separation

Dataset provenance is kept separate from adaptation, validation and test
assignment.

Patient-level partitioning and nested few-shot support sets remain governed by
the leakage-safe experimental protocol.