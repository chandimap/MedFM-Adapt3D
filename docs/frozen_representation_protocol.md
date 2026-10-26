# Audited Frozen Volumetric Representation

## Scientific Question and Supported Inference

Can the fixed features of one released 3D medical encoder support an aligned
pulmonary-nodule segmentation head while the encoder remains unchanged?
The task is the existing oracle-centred, reader-specific localized segmentation
task. Reference-derived centres are supplied. This is not whole-scan detection,
screening, malignancy diagnosis, a clinical device, or a claim of clinical readiness.

The integrated model is **SuPreM SegResNet, 2100-CT release**, identified by
Hugging Face revision `46ab78c25b07ace573177ebffd02214f670c7e4f` and SHA-256
`2db81dc05cd9ea7234ca75e921e53e32b8716dc4cba88a6710742bfc282589a3`.
The public repository headline describes a larger training collection; it does
not establish the training population of this checkpoint. The paper also reports
validation-based model selection, which must be considered when assessing exposure.

The source-specific findings and immutable upstream code revision are recorded in
[`suprem_segresnet.yaml`](../configs/foundation/suprem_segresnet.yaml).
The checkpoint checksum and size were independently matched to the downloaded
official release during implementation verification. A content hash establishes
byte identity, not dataset or patient independence.

## Provenance Decision

Patient-level disjointness from the local LIDC-IDRI evaluation population is
**NOT_ESTABLISHED**. Dataset names, anatomy, and the absence of an explicit LIDC
entry are insufficient proof. An exact checkpoint-to-training/selection-population
crosswalk and a defensible local patient/source linkage audit were not available
in this implementation verification.

This integration therefore permits synthetic verification and train/validation
optimization evidence only. The execution path has no definitive-test phase;
`RunAccess` is instantiated with the calibration barrier even during smoke runs.
The clean-generalization and clinical-readiness claim gates reject requests.
These are project API safeguards, not encryption or protection against a researcher
modifying the Python code or opening source files directly.

The published VoCo 10k recipe explicitly includes LIDC and LUNA16. It is not
selected for a clean LIDC-IDRI generalization comparison. This finding concerns
that identified release, rather than asserting the provenance of every VoCo variant.

## Representation and head

The encoder uses MONAI SegResNet: one input channel, initial width 16, encoder
blocks `(1,2,2,4)`, GroupNorm with eight groups, and no active dropout. All 58
encoder tensors are required, with exact keys, shapes and dtypes and finite
values. They contain 4,399,344 parameters. Decoder tensors are explicitly
discarded and their names recorded. No partial-load or random-weight fallback exists.

Each of four feature scales (16, 32, 64 and 128 channels) is projected to one
logit by a bias-free 1x1x1 convolution. The four logit maps are resized to the
padded input grid using trilinear interpolation with `align_corners=False`, then
summed with one shared bias. The head has `16+32+64+128+1 = 241` trainable
parameters. Interpolating logits avoids storing 240 full-resolution channels.
The high-resolution feature scale preserves access to local information that
would be lost by probing only the coarsest feature map of a small nodule.

Input dimensions are padded on the right by replication to multiples of eight.
The returned logits are cropped back to the original grid. Labels are neither
resampled nor interpolated inside the model. Tensor axes retain the existing
`[batch,channel,z,y,x]` convention. This defines the feature-to-logit resize rule;
it is not a new claim of subvoxel localization accuracy.

The encoder executes under `no_grad`, its parameters have `requires_grad=False`,
and `train()` keeps it in evaluation mode. Its complete state fingerprint is
checked at validation boundaries and after selection/reload. The optimizer
contains exactly the task head; encoder gradients are forbidden. Selected
checkpoints contain head tensors only.

## Controlled Comparison

`suprem_frozen` and `random_frozen` share the exact encoder architecture, head,
head initialization, candidate index, partitions, support patients, sampler,
augmentation seed, optimizer/loss, validation rule and step-budget recipe.
The random representation has a fixed, separate representation seed (887).
The head RNG is reset after encoder construction/loading so loading a checkpoint
cannot alter head initialization. Random-representation variability is not
estimated by this fixed control.

The fully trainable conventional SegResNet reference uses encoder blocks
`(1,2,2,2)` and a full decoder. It remains valuable as a task reference, but a
comparison against that model alone cannot isolate pretraining from architecture
and trainability. The matched frozen control isolates the representation source
under this probe protocol, subject to unresolved pretraining provenance.

Validation selects each head independently using the existing patient-macro
Dice rule and earliest-step tie handling. The same configured optimization
budget need not result in the same selected or stopping step. The paired
comparison checks shared conditions and reports a descriptive validation Dice
difference only if both optimization-adequacy gates passed. Otherwise it records
the two statuses and a null difference. Because validation selected the heads,
an emitted difference is not an
unbiased test estimate, a significance test, or proof of useful clinical transfer.

## Clinical and Input Assumptions

The existing candidate index and its sealed preprocessing trace are reused.
Real-data preparation retains the prespecified thoracic HU window [-1000,400],
output range [-1,1], LPS orientation, 1-mm isotropic spacing, 64-mm candidate
extent, reader-specific masks, and geometry QC. No hidden normalization,
anatomical reflection, image resampling, or consensus-mask collapse is introduced.

Published SuPreM training-code defaults describe a different abdominal window,
range and spacing. Those defaults do not prove the full recipe of the selected
release. This protocol deliberately fixes the common thoracic input to ask
about representation utility in the current target setting. Compatibility with
the source input recipe is **NOT_ESTABLISHED**. A negative result can reflect
input/domain mismatch, head capacity or optimization rather than an intrinsic
failure of pretraining. This implementation makes no performance claim either way.

Patients, rather than slices, nodules or reader masks, define K. Nested supports
and partitions are unchanged. Training reuses patient-balanced reader rotation
and the existing DiceCE objective. Validation reuses patient-macro Dice,
physical HD95 with explicit undefined values, volume error, sensitivity and
empty-prediction counts. Reader annotations remain dependent observations,
not independent patients. Phantom tests cannot establish clinical accuracy or
population uncertainty. The ten-patient synthetic smoke cannot satisfy K=10/20
after partitioning and must not be silently presented as those budgets.

## Inspectable Evidence and Publication Boundary

Each run exclusively creates provenance, per-step loss, a selected-head checkpoint,
a checkpoint lock, optimization diagnostics, a sealed result and completion hashes.
Failed execution leaves a failure marker when an output directory exists. The
reader verifies completion hashes before comparison. These hashes detect drift
relative to the receipt; they are not cryptographic signatures or a tamper-proof ledger.

Optimization adequacy is reported independently from successful software execution.
A smoke run can complete while its adequacy gate is blocked; no test is released
in either case. Calibration uses the smallest/largest feasible K and its separate
engineering head seed. An existing baseline test-release marker closes further
representation calibration on that candidate directory.

Private candidates, identifiers, weights and run outputs remain outside Git.
The public evidence file contains aggregate synthetic implementation checks and
explicitly identifies the official checkpoint used. It contains no pretrained
tensors, adapted encoder tensors, or medical records.

The upstream GitHub licence states CC BY-NC-ND 4.0 while the Hugging Face
metadata declares Apache-2.0. Both notices are recorded without asserting a
resolution of the conflicting terms. No upstream implementation is vendored;
the adapter uses the project's existing MONAI dependency. No upstream or adapted
encoder weights are included in this repository or its supplied patch.
