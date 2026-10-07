# Conventional random-initialized segmentation reference

## Research question and role

How much can a conventional randomly initialized 3D segmentation model learn
for oracle-centred localized pulmonary-nodule segmentation using only 1, 5, 10,
or 20 labelled patients, under the existing leakage-safe, reader-aware,
geometry-preserving protocol?

The scientific expectation is that additional labelled patients may improve
held-out patient-macro segmentation performance, with substantial variability
under scarcity. This is an empirical question, not an assumed monotonic result
or a statistical hypothesis test in this commit. The reference will support
later adaptation comparisons; it does not itself test a foundation model.

Commit 01 defines patient separation and nested support. Commit 02 defines
source lineage and reader observations. Commit 03 defines the physical CT/mask
transform. This milestone adds reference-model validity: balanced training,
an optimization opportunity reviewed without test labels, repeated random
initializations, physical evaluation, and inspectable immutable evidence.

## Task and target population

The task is **oracle-centred localized 3D pulmonary-nodule segmentation**.
Each eligible reader-specific contour supplies its own reference-mask centroid
through the existing rasterizer. Test crops also receive this oracle centre.
There is no automated lesion localization stage.

Targets are the existing LIDC XML >=3 mm nodule annotations carrying valid
characteristics and passing all existing QC. Patients without eligible targets
do not count toward K. Failed source/geometry QC aborts preparation. Matched
series without eligible contours and CT series lacking a matching XML in the
supplied source are recorded explicitly in the private source audit.

One candidate is one reader-specific observation, not necessarily a distinct
anatomical lesion. Cross-reader lesion identities are not inferred from local
nodule IDs. Multiple readers can yield different oracle centres/crops. This
protocol does not quantify agreement on a shared lesion grid or manufacture
consensus/pathology labels. All observations from one patient stay together.

## Patient budgets and randomness

Eligible patients are sorted and partitioned using a local RNG with seed 101.
Validation and test each receive `max(1, floor(0.2 * N))` patients; the remainder
is the training pool. At least three eligible patients are required. This is an
unstratified fixed partition, not a representative population sample.

The existing `validate_disjoint_patient_partitions` checks separation; the
existing `build_nested_support_sets` builds a single ordering using seed 211.
Support satisfies S1 ⊂ S5 ⊂ S10 ⊂ S20 wherever those budgets are feasible.
All eligible targets from each selected support patient are retained.

With 10 eligible pilot patients, the default split has 6 training, 2 validation,
and 2 test patients. K=1/5 are feasible; requesting K=10/20 raises an explicit
exception before training. The matrix preflights every requested K and output
directory. It does not silently shrink a request or skip existing runs.

| Source | Prespecified value | Role |
|---|---|---|
| Partition | 101 | Train/validation/test membership |
| Support | 211 | Nested support-patient order |
| Definitive initialization | 17, 29, 43, 59, 71 | Five independent model initializations |
| Augmentation | 307 | Separate reserved RNG; identity policy consumes no draws |
| Loader/order | 401 | Balanced patient and reader-target cycles |
| Calibration initialization | 997 | Separate engineering experiment |
| Smoke initialization | 991 | Software/data-path verification |

Augmentation and training-order schedules are fixed across initialization seeds.
This common-randomness design measures variability conditional on initialization
under a fixed cohort and training recipe. It does not characterize all possible
partition, support, augmentation, or order variability. GPU determinism is
requested, not assumed across different devices/software versions; unsupported
deterministic operations fail instead of silently disabling the requirement.

## Model, sampling, and optimization

The primary YAML declares a MONAI SegResNet with 16 initial filters, encoder
blocks [1,2,2,2], decoder blocks [1,1,1], GroupNorm (8 groups), and trainable
deconvolution. It has one binary foreground logit and 2,973,521 parameters under
the pinned MONAI version; the runtime audit is authoritative. All parameters
are trainable. There is no dropout or pretrained/checkpoint initialization.

The factory temporarily blocks supported PyTorch checkpoint/URL loading routes
during construction, fingerprints the complete initial state, and audits every
parameter using the existing parameter utility. Same-seed reproducibility and
different-seed initialization are tested. Later restoration of the run's own
validation-selected checkpoint is explicit and hash checked.

Commit-03 64 mm crops are 65 voxels per dimension at 1 mm spacing. SegResNet
internally pads on the right with replicated intensity to a multiple of eight,
then removes the padding from logits. Targets and stored geometry remain on
the original grid. No crop reduction or mask interpolation is introduced.

Patient sampling visits each support patient once per shuffled cycle. Within
that patient, all eligible reader targets rotate through shuffled cycles before
repeating. Thus annotation-rich patients receive more target diversity, not
more optimization weight. Batch size is fixed at two observations even at K=1;
exposure and unvisited targets are recorded. Loading uses one process without
worker randomness. The primary augmentation is identity: all spatial
reflections are disabled after LPS normalization. Anterior/posterior and
superior/inferior inversion are not assumed clinically plausible, and even
left/right reflection is omitted for this conservative reference. CT/target
alignment and binary targets remain exact. The separate augmentation RNG is
retained without consuming draws. Validation/test are also unaugmented. Later
comparison methods should receive the same opportunity where technically appropriate.

The primary optimization settings are AdamW, learning rate 0.001, weight decay
0.00001, equal-weight sigmoid Dice + BCE (`DiceCELoss`), gradient-norm clipping
at 5, at most 8,000 updates, at least 2,000 updates, validation every 200 updates,
and early stopping after 10 validations without an improvement exceeding
0.0005. The learning rate is constant. Checkpoint selection uses strictly best
patient-macro validation Dice, with the earliest step breaking exact ties;
`min_delta` controls stopping, not which checkpoint wins.

These are concrete engineering starting values, not a claim that convergence
has been demonstrated on LIDC. Step budgets keep optimizer opportunity and
observations per update consistent across K; equal epochs would not. Training
time and the number/diversity of exposed targets can still differ across K.
Full loss/validation trajectories and exposure counts must be reviewed locally.

## Calibration, freeze, and test firewall

1. A real-data smoke run performs 12 updates and validation only. It verifies
   candidate loading and execution; it is not a definitive experiment.
2. Calibration uses seed 997 and the smallest/largest feasible K, with train and
   validation only. Compare at most three documented global recipes; keep the
   other scientific assumptions fixed. Use the same recipe across K. Preserve
   every trial and record the train/validation rationale for the selection.
   `LockedProtocol.calibration_budgets` derives these extremes from locked
   support availability: 1/20 for a full cohort, 1/10 if 10 is largest, 1/5 for
   the ten-patient pilot, and only 1 if no other budget is feasible. This checks
   one common recipe across scarcity extremes; it is not K-specific tuning.
3. Freeze requires completed current-recipe calibration at every feasible
   calibration budget, minimum updates, a patience plateau, non-collapsed
   validation predictions, and exposure of every support target. If the step
   cap was reached before a plateau, review/extend the budget using train/val
   only and rerun under a new output root. A plateau is an engineering criterion,
   not proof of a global optimum or adequate clinical performance.
4. The receipt binds configuration, package source snapshot, cohort index,
   preprocessing, and patient/support assignments. A changed recipe/source/cohort
   invalidates it. No test metrics can be passed into checkpoint selection.
5. Each definitive run trains, selects by validation, writes a locked checkpoint,
   restores only that checkpoint, completes final validation, and writes sealed
   `optimization_gate.json` BEFORE any real test release. Approval requires
   minimum updates, finite loss/gradients/updated weights/logits, complete
   patient/reader exposure, balanced patient contribution, and a patience
   plateau. All-empty/all-full validation predictions or zero validation overlap
   block release. At the cap, fewer than the prescribed stale validations is
   inadequate even if the most recent point did not improve. Test stays closed;
   the attempt receives FAILED evidence instead of test scores. Diagnostics use
   training/validation only and are also retained for engineering review.
   The selected validation Dice must reproduce; model and checkpoint fingerprints
   must remain fixed through test evaluation. Source/configuration/receipt and
   on-disk partition/index bindings are checked again before test can open once.
6. First definitive test access records a private cohort release ledger. Further
   calibration/freezing on that cohort is rejected. All later runs must share
   the same frozen receipt. Retuning after seeing definitive test results requires
   a new study and fresh held-out patients; do not delete the ledger to bypass it.

This revision uses the permitted **per-run barrier**, preserving the existing
single-run artifact lifecycle rather than adding a cohort-wide two-stage engine.
The matrix is sequential: an adequate run may be evaluated before another run
is trained. If a later run is inadequate after any test exposure, stop: the
recipe cannot be revised on that cohort, and a partial matrix is not a complete
baseline. Default reporting refuses missing seeds. Calibration at both extremes
reduces this risk but does not prove every initialization will converge. There
is no guarantee of a global optimum or clinical adequacy.

## Input metadata and restricted preparation QC

Candidate index schema 1.1 exposes patient/series/reader identity, opaque CT/mask
and QC references/hashes, CT geometry/intensity trace, preprocessing policy,
source lineage and the declared oracle centre. Strict field checks reject
additional target-shape quantities. The centre and target eligibility are
label-derived task inputs; localization is not independent of the target.

Full unchanged Commit-03 mask-preservation evidence (source/resampled/cropped
volumes, voxel counts, centroids and QC ratios) lives in separately sealed
`target_qc/<candidate_id>.json` files. It is not in `CandidateRecord.trace` or
the training-visible index payload. `RunAccess` authorizes target-array/QC loads;
real smoke/calibration cannot open held-out target arrays or parse that QC.
Authorized loads verify QC, hash/lineage/grid binding and cropped foreground
count. Opaque integrity checks can hash bytes without decoding labels or QC.
Preparation retains every existing fail-closed geometry gate.

Old schema-1.0 indexes contain excessive target metadata and are rejected.
Prepare fresh candidates/partitions; retain old private artifacts for audit.
Sealed files are integrity-bound, not encrypted. Low-level loaders and direct
filesystem access are preparation/audit facilities; training must use RunAccess.

The firewall constrains the project execution path. It cannot prevent a person
from reading local files outside it, editing code, or forging every audit file.
Hashes are integrity checks, not digital signatures. Scientific validity also
depends on correctly reconciled source patient identity and honest data curation.
Smoke/calibration never open real test target arrays; synthetic smoke fixtures
exercise final test access without releasing clinical labels.

## Evaluation and repeated-seed reporting

Each target is evaluated without augmentation at the frozen probability
threshold (default 0.5). The primary grid is the audited preprocessed candidate
grid, not an unverified original-space inverse transform.

| Metric | Definition and failure handling |
|---|---|
| Dice | Binary overlap; empty prediction gives 0 |
| HD95 (mm) | Maximum of directed 95th-percentile distances between six-connected boundary voxel centres; exact Euclidean physical distances using actual XYZ spacing |
| Absolute volume error (mm³) | Absolute foreground-voxel-count difference × physical voxel volume |
| Reference-voxel sensitivity | Fraction of reference foreground predicted foreground; empty prediction gives 0 |
| Empty prediction | Explicit observation count and equally weighted patient-macro rate |

Padding the boundary calculation with background includes masks touching the
crop edge. An empty prediction has undefined HD95 (`null`), never zero or a
made-up cap. Primary patient/cohort HD95 is undefined if any contributing
observation is undefined. Separately named conditional HD95 and completeness
counts are available; do not rank methods on defined-only distances while
ignoring empty failures.

Observation values are averaged within patient, then equally across patients.
This preserves the patient statistical unit despite correlated reader targets.
Each individual seed remains inspectable. Each feasible K reports all five
values, mean, sample standard deviation (ddof=1), median, linear-percentile IQR,
minimum and maximum where defined. If a metric has undefined seed values,
primary statistics remain null and any defined-only statistics are separately
labelled. Missing seeds fail summary generation unless explicitly requested as
an incomplete engineering summary.

`label_efficiency_seeds.csv` contains each K/seed/partition value and failure
counts for future curves and method comparisons. No best-seed filtering,
patient confidence interval, hypothesis testing, or population inference is
implemented. A tiny held-out pilot cannot support population-level conclusions.

## Provenance and artifact privacy

Local artifacts record software/CUDA/device details, Git SHA/dirty state,
initial/selected model fingerprints, full parameter accounting, all RNG roles,
support and held-out assignments, configuration/source/index fingerprints,
optimizer/loss/threshold, actual updates and selected step, validation/test
metrics, reader/patient counts, and training exposure. Each completed run has
`provenance.json`, `training.jsonl`, `checkpoint_lock.json`, `optimization_gate.json`,
`selected_checkpoint.pt`, `result.json`, and `COMPLETED.json`; failures receive
`FAILED.json`. Completed directories are never overwritten or automatically
resumed. Keep failed attempts for audit; select a new root for a rerun.

Private indexes/UIDs, arrays, checkpoints, and all runs remain gitignored.
Publish only carefully reviewed non-sensitive aggregate evidence after the
real-data gate. Package configuration/protocol/code belongs in Git; patient
assignments and results do not. CI checks invariants and synthetic execution,
not the private 20-run experimental matrix.

## Interpretation limits

- Supplied reference centres exclude whole-scan detection and autonomous localization claims.
- Reader annotations are correlated references, not independent patients or pathology truth.
- No screening, malignancy diagnosis, outcome prediction, clinical deployment, or clinically sufficient K claim.
- No measured real LIDC baseline performance is included with this implementation.
- Initialization variability under a fixed cohort is not population uncertainty.
- Repeated model seeds quantify optimization and initialization variability
  conditional on one fixed patient partition and one fixed nested support ordering.
  They do not quantify uncertainty due to alternative cohort partitions or
  alternative K-patient support samples, or establish population generalizability.
- A different-architecture foundation-model comparison cannot isolate pretraining causally.
- Architecture-controlled pretraining ablations, independent domains, calibration,
  uncertainty decomposition, and agentic AI remain later research milestones.

An accurate current statement is: **we implement and synthetically verify an
auditable, leakage-safe, reader-aware conventional random-initialized 3D
segmentation reference protocol.** Replace “implement and synthetically verify”
with a completed experimental claim only after the documented local LIDC runs
and all prespecified seed results actually exist.
