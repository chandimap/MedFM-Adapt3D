# Final Commit 04 verification

Verification date: **2026-10-06**. Reviewed Commit-03 base:
`48e9269b87b3824ac440c2ce61914ff6cc7e39c0`.
This final revision updates the exact latest **30-file** source overlay, without
returning to the earlier implementation. All 45 other tracked base files are
byte-for-byte unchanged. No private LIDC data was available; all verification
images, XML, masks, score fixtures and training inputs are explicitly synthetic.

## Observed execution matrix

| Check | Result | Measured evidence / limit |
|---|---|---|
| Clean Python environment and editable install | PASS | Fresh venv; pinned CPU PyTorch; `pip install -e '.[dev]' build` |
| `python -m pip check` | PASS | No broken requirements |
| `python -m ruff check .` | PASS | Ruff 0.16.10; all checks passed |
| `python -m compileall -q src scripts tests` | PASS | All source, scripts and tests compile |
| Complete `python -m pytest -q` | PASS | **112 passed in 135.81 s**; original 33 plus 79 baseline cases |
| Existing research-contract validator | PASS | Existing nested budgets, lineage and parameter-policy checks |
| Existing preprocessing-contract validator | PASS | 65³ grid; tiny-nodule tolerance; crop retention 1.0; centroid shift 0 mm |
| Revised random-baseline validator | PASS | **21 named gates**, including target metadata and pre-test adequacy |
| Synthetic DICOM/XML preparation | PASS | Actual Commit-02 reconciliation and Commit-03 HU/reader/preprocessing path |
| Standalone synthetic end-to-end | PASS | Actual SegResNet/AdamW; 12 updates; selected step 12; lock and final validation/test artifacts |
| Real-data POLICY integration on synthetic arrays | PASS | Adequate definitive path; cap-improvement/numerical/drift failures before test; no real LIDC inputs |
| Determinism and random initialization | PASS | Same-seed training reproduced; five actual initial-state fingerprints differ; all 2,973,521 parameters trainable |
| Full real crop geometry | PASS | Actual 65³ forward/backward; aligned logits and gradients |
| Calibration at feasible extremes | PASS | 1, 1/5, 1/10, 1/20 availability fixtures; one-recipe freeze; per-K recipe mixture rejected |
| All-seed reporting | PASS | All 20 prescribed pairs preserved in synthetic reporting fixtures; missing seeds fail by default |
| Wheel and source-distribution build | PASS | Both build; source retains configuration, CI, scripts, docs and fixtures |
| Extracted source-distribution validator | PASS | All 21 gates execute using the extracted package source/configuration |
| Privacy/source-package audit | PASS | No arrays/XML/DICOM/weights/runs/caches/environments; 14 ignore probes pass; 45 protected base files unchanged |
| CI source audit | PASS | Dependency, Ruff, compilation, full pytest, all three validators and synthetic execution present |
| Genuine private LIDC preparation/smoke | **NOT RUN** | Mandatory local release gate before merge |
| Real-data optimization adequacy / calibration | **NOT RUN** | Engineering defaults require genuine train/validation evidence |
| Definitive repeated-seed LIDC performance | **NOT RUN** | No real learning curve or patient performance was measured |
| CUDA training / determinism | **NOT RUN** | CPU-only runtime; CUDA unavailable |
| Native Windows / PowerShell execution | **NOT RUN** | Detailed commands supplied; actual environment is Linux |
| Hosted GitHub Actions | **NOT RUN** | Local equivalents pass; inspect the actual latest PR checks before merging |

Runtime: Python **3.12.14**, PyTorch **2.13.0+cpu**, MONAI **1.6.0**,
NumPy **2.5.3**, pydicom **3.0.2**, SimpleITK **2.5.6**, pytest **9.1.1**.
Linux x86_64, glibc 2.39; `torch.cuda.is_available()` is false.
The actual factory reports 2,973,521 total and trainable parameters, fraction 1.0,
random initialization and no construction-time weight loading.

The first fresh full-suite pass had 111 passes and one test-fixture failure:
a threshold-drift test changed 0.5 to 0.5 after the fixture was restored to the
primary threshold. The adversarial edit was corrected to 0.6; its rejection
assertion was retained. The complete verification was then rerun successfully.
An initial package-audit helper also expected `.gitignore` inside the source
distribution; that local Git-policy file is audited separately. The final ZIP
contains it, and the sdist contains all 29 required package resources.
No production safeguard or rejection assertion was weakened to obtain a pass.

## Revised scientific gates and evidence

| Requirement | Actual source / adversarial evidence |
|---|---|
| Cap with continuing improvement cannot expose test | Actual training with injected increasing validation trajectory; blocked `optimization_gate.json`, FAILED, zero test loads and no cohort release ledger |
| Finite training before release | Loss and gradient failure injections; actual loop also checks updated parameters and evaluation logits |
| Complete lock/final validation/minimum steps/exposure | Eleven failure-field cases; shared adequacy helper used by freeze and definitive approval |
| Obvious validation collapse blocked | All-empty, all-full and zero-overlap flags rejected before definitive test |
| Adequate frozen run can test once | Actual tiny real-policy run; lock → final validation → adequacy → once-only arrays/QC → completion |
| Late recipe/source/cohort drift blocked | Receipt, source, sealed index and partition edits during final validation rejected before held-out access |
| Target-shape metadata restricted | Input-only whitelist; target-statistic injection at record/trace/grid/audit levels rejected even after re-sealing |
| Original preprocessing QC retained | Full Commit-03 mask trace saved separately in sealed `target_qc`; authorized loads recheck QC and actual foreground count |
| Integrity without content disclosure | Image/mask/QC bytes hashed while array/QC decoders are forbidden |
| Engineering test masks/QC remain closed | Real-policy smoke/calibration succeed with deliberately unreadable held-out mask AND QC files; project access rejects reads |
| Oracle-centre distinction | Centre retained explicitly as label-derived task input; no claim of target-independent localization |
| Extreme-budget calibration | Locked supports determine min/max feasible K, unique if equal; full cohort uses 1/20 |
| One common recipe | Freeze requires current configuration at both extremes; mixed per-K recipes fail |
| Conservative augmentation | Identity policy; nonzero reflections rejected; values, orientation and augmentation RNG state unchanged |
| Original scarcity/reader/physical contracts | Patient-disjoint/nested and seed-invariant membership; balanced cycles/all-reader rotation; macro weighting; anisotropic mm/mm³ metrics and honest empty/undefined cases |
| Five-seed reporting | Every seed retained; no best-seed filter; default complete reporting refuses missing seeds |

Most integration tests execute the real model, optimizer, preprocessing, loader,
selection and artifact code. Explicit increasing scores/non-finite injections
are adversarial inputs. Reporting-only tests stub storage decoding for synthetic
score fixtures. Some validator state-machine probes use a labelled synthetic
marker checkpoint and analytic validation inputs; these do not claim trained
weights or clinical results. Separate integration tests execute actual training.

## Precise release and interpretation boundaries

The final implementation uses a **per-run** adequacy barrier, preserving the
existing artifact lifecycle. It does not implement a cohort-wide all-models-ready
Stage A/B barrier. An adequate earlier run can test before a later run trains.
Once any definitive test opens, the cohort ledger forbids later calibration or
recipe changes. If a later run fails adequacy, retain the incomplete study; do
not retune against that test cohort. A revised recipe needs a new held-out study.
Default summaries reject incomplete prescribed matrices.

Candidate indexes require schema **1.1**. Reprepare private candidates/partitions
through the unchanged Commit-02/03 pipeline; retain old outputs without editing
their schema. Input metadata includes CT geometry, lineage, opaque hashes and the
annotation-derived oracle centre. Preparation eligibility is label-informed.
Additional held-out shape/QC content is restricted until authorized target loads.
The barrier protects the project execution path, not arbitrary filesystem/Python
access or malicious re-signing of every hash.

All previous CI gates remain. Explicit compilation now joins dependency checks,
Ruff, full pytest, both existing validators, the revised baseline validator and
synthetic optimization/evaluation. No private dataset or full real 20-run matrix
is executed in CI. No runtime dependency, earlier preprocessing setting, or
patient-support implementation was replaced.

Repeated model seeds quantify initialization/optimization variability conditional
on **one fixed patient partition and one fixed nested support ordering**. They do
not quantify alternative cohort partitions, alternative K-patient support samples
or clinical population uncertainty. Oracle-centred segmentation is not detection,
screening, malignancy diagnosis, outcome prediction or clinical deployment.
No clinically sufficient K or population generalizability is established.
A later different-architecture foundation comparison cannot attribute differences
solely to pretraining.

This is software and scientific-contract evidence. Genuine local LIDC smoke
training/validation, privacy review and every hosted PR check remain mandatory
before merging; real convergence and definitive performance remain unmeasured.
