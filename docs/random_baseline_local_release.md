# Windows, VS Code, and PowerShell local release gate

Use **01-Commit04-FINAL-Code-Only.zip**, the final 30-file source overlay based on
main `48e9269b87b3824ac440c2ce61914ff6cc7e39c0` (Commits 01–03).
The separate **02-Commit04-FINAL-VS-Code-Steps.docx** is the detailed starting guide.
These instructions use the same final source and scientific gates; no patch is needed.
If the previous 30-file version is already applied, use the checked update route
in that Word guide. Preserve unrelated work and all ignored private data.

The minimum merge gate is every software/scientific check plus genuine local
LIDC smoke training/validation at feasible budgets. Calibration, freeze and the
five-seed definitive study are later experiments. Synthetic evidence never
establishes real LIDC performance. Candidate schema 1.1 requires fresh private
candidate/partition preparation; retain old private outputs, do not edit their schema.

Run from the repository root in VS Code's PowerShell terminal. Select
`.venv\Scripts\python.exe` as the VS Code interpreter; commands call it directly.

## 1. Extract the final source and create the branch

Download the final code-only ZIP outside the repository. The default route below
is for a clean Commits 01–03 checkout; the Word guide also covers an already-applied
previous 30-file version. Verify the ZIP SHA256 recorded in that guide before copying.

```powershell
$RepoPath = 'D:\GITHUB\PROJECTS\AI-RESEARCH\MedFM-Adapt3D'
$BundleDir = Join-Path $env:USERPROFILE 'Downloads\Commit04-FINAL-code'
$ZipPath = Join-Path $env:USERPROFILE 'Downloads\01-Commit04-FINAL-Code-Only.zip'
if (Test-Path $BundleDir) { throw 'Use a fresh extraction folder.' }
Expand-Archive -LiteralPath $ZipPath -DestinationPath $BundleDir -ErrorAction Stop
Set-Location $RepoPath
function Assert-Passed([string]$StepName) {
    if ($LASTEXITCODE -ne 0) { throw "$StepName failed. Stop and inspect the error." }
}
$PendingChanges = git status --porcelain
Assert-Passed 'Check working tree'
if ($PendingChanges) { throw 'Use the checked update route for prior source; preserve other work.' }
git fetch --prune origin
Assert-Passed 'Fetch'
git switch main
Assert-Passed 'Switch main'
git pull --ff-only origin main
Assert-Passed 'Update main'
$CurrentBase = git rev-parse HEAD
Assert-Passed 'Read base SHA'
if ($CurrentBase.Trim() -ne '48e9269b87b3824ac440c2ce61914ff6cc7e39c0') {
    throw 'main differs from the reviewed base; review compatibility first.'
}
$BranchName = 'research/random-segresnet-reference-final'
git switch -c $BranchName
Assert-Passed 'Create final reference branch'
```

Expected: reviewed SHA, feature branch, no unrelated changes. Do not force-reset,
change source safeguards, or overwrite a different upstream implementation.

## 2. Copy only the final 30 repository files

```powershell
$SourceFiles = @(
    '.github/workflows/ci.yml'
    '.gitignore'
    'MANIFEST.in'
    'README.md'
    'configs/baseline/random_segresnet.yaml'
    'docs/random_baseline_local_release.md'
    'docs/random_baseline_protocol.md'
    'docs/random_baseline_verification.md'
    'scripts/smoke_random_baseline.py'
    'scripts/validate_random_baseline_contract.py'
    'src/medfm_adapt3d/baseline/__init__.py'
    'src/medfm_adapt3d/baseline/artifacts.py'
    'src/medfm_adapt3d/baseline/candidates.py'
    'src/medfm_adapt3d/baseline/cli.py'
    'src/medfm_adapt3d/baseline/config.py'
    'src/medfm_adapt3d/baseline/metrics.py'
    'src/medfm_adapt3d/baseline/model.py'
    'src/medfm_adapt3d/baseline/protocol.py'
    'src/medfm_adapt3d/baseline/reporting.py'
    'src/medfm_adapt3d/baseline/sampling.py'
    'src/medfm_adapt3d/baseline/selection.py'
    'src/medfm_adapt3d/baseline/synthetic.py'
    'src/medfm_adapt3d/baseline/training.py'
    'tests/conftest.py'
    'tests/test_random_baseline_lidc_preparation.py'
    'tests/test_random_baseline_metrics.py'
    'tests/test_random_baseline_model_sampling.py'
    'tests/test_random_baseline_protocol.py'
    'tests/test_random_baseline_reporting.py'
    'tests/test_random_baseline_training.py'
)
$ModifiedFiles = @('.github/workflows/ci.yml', '.gitignore', 'README.md')
$Extracted = @(Get-ChildItem -LiteralPath $BundleDir -File -Recurse -Force)
if ($Extracted.Count -ne 30) { throw 'Expected exactly 30 source files.' }
foreach ($Relative in $SourceFiles) {
    $From = Join-Path $BundleDir $Relative
    $To = Join-Path $RepoPath $Relative
    if (!(Test-Path -LiteralPath $From -PathType Leaf)) { throw "Missing $Relative" }
    if ((Test-Path -LiteralPath $To) -and $Relative -notin $ModifiedFiles) {
        throw "New file already exists: $Relative; use the checked update route."
    }
}
foreach ($Relative in $SourceFiles) {
    $From = Join-Path $BundleDir $Relative
    $To = Join-Path $RepoPath $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $To) -Force | Out-Null
    Copy-Item -LiteralPath $From -Destination $To -Force -ErrorAction Stop
    if ((Get-FileHash $From).Hash -ne (Get-FileHash $To).Hash) {
        throw "Copy integrity failure: $Relative"
    }
}
git status --short
Assert-Passed 'Inspect installed source'
```

Expected: 3 modified and 27 new files relative to the reviewed base. ZIP entries
are repository-relative; do not create a second nested project folder. Teaching
guides, the ZIP, private arrays/QC/indexes and run outputs stay outside staging.

## 3. Install or update the existing Python 3.12 environment

```powershell
if (!(Test-Path '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    Assert-Passed 'Create Python 3.12 environment'
}
$PythonExe = Join-Path $RepoPath '.venv\Scripts\python.exe'
& $PythonExe --version
Assert-Passed 'Read Python version'
& $PythonExe -m pip install --upgrade pip
Assert-Passed 'Update pip'
& $PythonExe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
Assert-Passed 'Install pinned CPU PyTorch'
& $PythonExe -m pip install -e '.[dev]' build
Assert-Passed 'Install project'
& $PythonExe -m pip check
Assert-Passed 'Check dependencies'
```

Expected: Python 3.12.x, PyTorch 2.13.0 (CPU build), MONAI 1.6.0, and “No broken
requirements found.” Preserve the pinned versions. CPU is sufficient for the
merge smoke gate. A compatible GPU build of the same pinned PyTorch can be used
for later definitive experiments; the supplied environment has not verified CUDA.

## 4. Run Ruff and compilation

```powershell
& $PythonExe -m ruff check .
Assert-Passed 'Ruff'
& $PythonExe -m compileall -q src scripts tests
Assert-Passed 'Python compilation'
```

Expected: “All checks passed!” and no compilation errors. Correct any failure
before proceeding; do not suppress a scientific check to obtain a green result.

## 5. Run the complete test suite

```powershell
& $PythonExe -m pytest -q
Assert-Passed 'Full pytest suite'
```

Expected: the full test count recorded in `random_baseline_verification.md`,
with no failed tests. These include original safeguards and new adversarial,
physical-metric, artifact, reader-balance, and actual synthetic training paths.

## 6. Run both existing scientific validators

```powershell
& $PythonExe scripts/validate_research_contract.py
Assert-Passed 'Existing research contract'
& $PythonExe scripts/validate_preprocessing_contract.py
Assert-Passed 'Existing preprocessing contract'
```

Expected: both contract PASS lines, nested 1/5/10/20 supports, and the existing
65³ preprocessing output with passing tiny-nodule preservation evidence.

## 7. Run the new scientific validator and synthetic execution

```powershell
$AuditTag = Get-Date -Format 'yyyyMMdd-HHmmss'
& $PythonExe scripts/validate_random_baseline_contract.py --report "outputs/baseline-contract-$AuditTag.json"
Assert-Passed 'Random baseline contract'
& $PythonExe scripts/smoke_random_baseline.py --output "outputs/baseline-synthetic-$AuditTag"
Assert-Passed 'Synthetic end-to-end execution'
& $PythonExe -m build
Assert-Passed 'Build wheel and source distribution'
```

Expected: all individual PASS lines and the final contract PASS, followed by a
synthetic end-to-end PASS and artifact directory. These are software evidence,
not real LIDC performance. Existing output directories intentionally cause failure;
use a new tag for a rerun and preserve prior evidence.
The build must produce both a wheel and a source archive under ignored `dist/`.
The source archive retains configs, scripts, documentation, and test fixtures.

## 8. Prepare genuine local LIDC candidates

Set the two source paths to your actual local folders. The XML root must contain
the verified final annotation files for your DICOM subset. If two XML files map
to the same locally available study/series, existing reconciliation rejects the
ambiguity. Resolve it using the Commit-02 cohort protocol and source provenance;
do not pick a file merely because it produces better model results.

```powershell
$DicomRoot = 'D:\DATA\LIDC-IDRI-pilot'
$XmlRoot = 'D:\DATA\LIDC-IDRI-XML'
$CandidateDir = "data/processed/random-baseline-$AuditTag"
& $PythonExe -m medfm_adapt3d.baseline.cli prepare --dicom-root $DicomRoot --xml-root $XmlRoot --output $CandidateDir
Assert-Passed 'Real LIDC preparation'
$IndexPath = "$CandidateDir/index.json"
$PartitionPath = "$CandidateDir/partitions.json"
$BaselineConfig = 'configs/baseline/random_segresnet.yaml'
```

Expected: eligible patient/reader-target counts and a private index. Preparation
uses the existing source/HU/reader/geometry safeguards and the unchanged 64 mm
preprocessing protocol. It aborts on invalid geometry, UID ambiguity, corrupted
data, or no eligible targets. The source audit records unmatched CT series and
matched series without eligible contours. Preparation failure leaves an audit
attempt directory; correct the source problem and choose a fresh candidate path.

## 9. Lock partitions and inspect the available train-patient count

```powershell
& $PythonExe -m medfm_adapt3d.baseline.cli partition --config $BaselineConfig --index $IndexPath --partitions $PartitionPath
Assert-Passed 'Create immutable patient partitions'
& $PythonExe -m medfm_adapt3d.baseline.cli inspect --config $BaselineConfig --index $IndexPath --partitions $PartitionPath
Assert-Passed 'Inspect label-budget feasibility'
$PartitionInfo = Get-Content $PartitionPath -Raw | ConvertFrom-Json
$FeasibleBudgets = @($PartitionInfo.supports.PSObject.Properties.Name | ForEach-Object { [int]$_ } | Sort-Object)
$UnavailableBudgets = @($PartitionInfo.unavailable_budgets)
Write-Host "Explicit smoke budgets: $FeasibleBudgets"
Write-Host "Unavailable budgets: $UnavailableBudgets"
```

Expected for 10 eligible patients: train=6, validation=2, test=2; feasible K=1/5,
unavailable K=10/20. Counts can differ if candidates are ineligible. K counts
patients, regardless of annotation counts. Requesting an unavailable K must
raise `Requested K=... but only ... eligible train patients`; this is the expected
scientific refusal, not an error to bypass. Do not reduce the test/validation
cohort merely to obtain a desired K after seeing data or performance.

## 10. Run genuine real-data smoke training and validation at feasible budgets

```powershell
$SmokeRoot = "runs/random_segresnet/release-smoke-$AuditTag"
foreach ($K in $FeasibleBudgets) {
    & $PythonExe -m medfm_adapt3d.baseline.cli run --config $BaselineConfig --index $IndexPath --partitions $PartitionPath --phase smoke --model-seed 991 --k $K --device cpu --threads 2 --output-root $SmokeRoot
    Assert-Passed "Real LIDC smoke K=$K"
}
```

Expected: each explicitly feasible run completes with a validation-selected
checkpoint, actual optimizer updates, validation metrics, and provenance.
These runs leave the real test labels sealed and do not satisfy the five-seed
definitive protocol. Poor/empty predictions after a few smoke updates are
possible and are represented honestly; inspect them without claiming efficacy.

## 11. Inspect real result and provenance artifacts

```powershell
foreach ($K in $FeasibleBudgets) {
    $RunDir = Join-Path $SmokeRoot ('smoke/k{0:D2}/seed991' -f $K)
    $Result = Get-Content (Join-Path $RunDir 'result.json') -Raw | ConvertFrom-Json
    $Provenance = Get-Content (Join-Path $RunDir 'provenance.json') -Raw | ConvertFrom-Json
    if (!(Test-Path (Join-Path $RunDir 'COMPLETED.json'))) { throw 'Run incomplete.' }
    if (Test-Path (Join-Path $RunDir 'FAILED.json')) { throw 'Run failed.' }
    if ($Result.synthetic -or $Result.test -ne $null) { throw 'Wrong smoke data/test policy.' }
    if ($Result.k_patients -ne $K -or @($Provenance.support_patients).Count -ne $K) { throw 'Invalid K.' }
    if ($Result.actual_optimization_steps -lt 4) { throw 'Too few smoke optimization updates.' }
    if ($Provenance.model.fraction_trainable -ne 1) { throw 'Baseline parameters are frozen.' }
    if ($Provenance.model.initialization -ne 'random') { throw 'Invalid initialization.' }
    if ($Result.test_access.read_counts.test -ne 0) { throw 'Real test labels were exposed.' }
    Write-Host "K=$K actual steps=$($Result.actual_optimization_steps) selected step=$($Result.selected_step)"
    $Result.validation.patient_macro | Format-List
    $Result.validation.counts | Format-List
    $Result.patient_training_exposure | Format-List
    $Provenance.runtime | Format-List
}
```

Inspect `optimization_gate.json`, `training.jsonl`, selected step, validation history, empty/undefined
counts, source/index/config fingerprints, patient exposure, and parameter counts.
Undefined HD95 is allowed when predictions are empty; treating it as zero is not.
The dirty Git state is expected before committing and must be recorded honestly.
Keep private IDs/UIDs and individual metric artifacts out of public screenshots/PR text.

## 12. Confirm private artifacts are ignored and nothing is staged

```powershell
git check-ignore -- $IndexPath $PartitionPath $SmokeRoot
Assert-Passed 'Check private artifact ignores'
git status --short
git diff --cached --name-only
```

Expected: private paths are ignored and no staged files exist yet. Raw DICOM/XML,
candidate arrays, full target-QC artifacts, private indexes, checkpoints, runs, caches, and virtual
environments must not appear as staged/new source. `.gitignore` is not a substitute
for reviewing the exact staging list, especially for extensionless raw data.

## 13. Stage only the supplied source allowlist

```powershell
git add -- $SourceFiles
Assert-Passed 'Stage intended source files'
$StagedFiles = @(git diff --cached --name-only)
Assert-Passed 'Read staged paths'
$UnexpectedFiles = @($StagedFiles | Where-Object { $_ -notin $SourceFiles })
if ($UnexpectedFiles.Count -gt 0) { throw "Unexpected staged paths: $UnexpectedFiles" }
```

Expected: only the marked new/modified files in the handoff. Do not use `git add .`
for this release. If anything private is staged, unstage that exact path, keep
the local file, correct ignores/locations, and review again.

## 14. Review the complete staged diff

```powershell
git diff --cached --check
Assert-Passed 'Whitespace/diff check'
git diff --cached --stat
git diff --cached
```

Confirm previous source safeguards and CI gates are retained; only intended
changes are staged; docs accurately say implementation/synthetic verification;
no patient identifier, local source-data path, result, or checkpoint is included.
The protocol/configuration/CI/test changes must all be present.

## 15. Commit with the professional message below

Save the exact message below to a UTF-8 text file outside the repository, then:

```powershell
$MessageFile = Join-Path $env:USERPROFILE 'Downloads\Commit04-FINAL-message.txt'
git commit -F $MessageFile
Assert-Passed 'Commit reviewed source'
git status --short
```

Expected: one source commit and a clean working tree (ignored private artifacts
remain local). The supplied message deliberately makes no real-performance claim.

## 16. Push the feature branch

```powershell
git push -u origin $BranchName
Assert-Passed 'Push feature branch'
```

Expected: the feature branch is pushed and linked to its remote tracking branch.
Do not push changes directly to `main`.

## 17. Open and review the pull request

Open [the repository pull-request page](https://github.com/chandimap/MedFM-Adapt3D/pulls),
choose **New pull request**, base `main`, compare your feature branch, and verify
the files against the handoff. Use the title and description below. Add the
local LIDC smoke counts and feasibility status only after actually observing
them; omit private IDs and avoid presenting smoke metrics as definitive results.

## 18. Check every GitHub Actions step

In the PR's **Checks** tab, open `CI / quality`. Expected successful sequence:
installation → pip check → Ruff → explicit compilation → complete pytest → existing research validator
→ existing preprocessing validator → new random-baseline validator → synthetic
end-to-end execution. Confirm it ran on the PR's latest commit SHA, not an older
green commit. A skipped, missing, cancelled, or failing required gate is not a pass.

## 19. Merge only after the complete gate passes

Merge after the local real-data smoke gate, source/privacy review, and every
required GitHub check pass. Prefer **Create a merge commit** to preserve the
reviewed feature history. If branch protection asks for additional checks/review,
satisfy them. Do not bypass or remove requirements to merge. After success,
GitHub's **Delete branch** button can remove the merged remote feature branch.

## 20. Synchronize local main safely

```powershell
git fetch --prune origin
Assert-Passed 'Fetch merged main'
git switch main
Assert-Passed 'Switch local main'
git pull --ff-only origin main
Assert-Passed 'Fast-forward merged main'
git status --short
git log -3 --oneline
git branch -d $BranchName
```

Expected: clean local main containing the merged change. If branch deletion is
refused after a squash/rebase merge, retain the branch until you have verified
the PR and source equivalence; do not force-delete an unreviewed branch.

## Later: calibration and the definitive five-seed matrix

Use the same candidate index and locked partitions. Run calibration on each
smallest/largest feasible K with seed 997 and the primary step budget. Preserve every trial.
Do not use smoke/test scores to optimize the recipe. If the cap is reached while
validation lacks a patience plateau or predictions are all empty/full or have zero overlap, review the
train/validation trajectories and adjust a global recipe, not one recipe per K.
Calibration receipts are deliberately refused when these diagnostics fail.

```powershell
$ExperimentTag = Get-Date -Format 'yyyyMMdd-HHmmss'
$ExperimentRoot = "runs/random_segresnet/experiment-$ExperimentTag"
$CalibrationBudgets = @(@($FeasibleBudgets[0], $FeasibleBudgets[-1]) | Sort-Object -Unique)
foreach ($K in $CalibrationBudgets) {
    & $PythonExe -m medfm_adapt3d.baseline.cli run --config $BaselineConfig --index $IndexPath --partitions $PartitionPath --phase calibration --model-seed 997 --k $K --device cpu --output-root $ExperimentRoot
    Assert-Passed "Calibration K=$K"
}
$FreezeArgs = @('--config', $BaselineConfig, '--index', $IndexPath, '--partitions', $PartitionPath,
    '--output', "$CandidateDir/frozen-recipe-$ExperimentTag.json",
    '--rationale', 'Reviewed train/validation loss and patient-macro Dice trajectories; global recipe fixed before test.')
foreach ($K in $CalibrationBudgets) {
    $CalibrationRun = Join-Path $ExperimentRoot ('calibration/k{0:D2}/seed997' -f $K)
    $FreezeArgs += @('--calibration-run', $CalibrationRun)
}
& $PythonExe -m medfm_adapt3d.baseline.cli freeze @FreezeArgs
Assert-Passed 'Freeze reviewed recipe'
$FrozenRecipe = "$CandidateDir/frozen-recipe-$ExperimentTag.json"
& $PythonExe -m medfm_adapt3d.baseline.cli matrix --config $BaselineConfig --index $IndexPath --partitions $PartitionPath --k $FeasibleBudgets --frozen-recipe $FrozenRecipe --device cpu --output-root $ExperimentRoot
Assert-Passed 'Complete definitive matrix'
& $PythonExe -m medfm_adapt3d.baseline.cli summarize --config $BaselineConfig --index $IndexPath --partitions $PartitionPath --frozen-recipe $FrozenRecipe --output-root $ExperimentRoot --output "$ExperimentRoot/summary"
Assert-Passed 'All-seed label-efficiency summary'
```

Replace the rationale with your actual documented train/validation review. If
you compared alternate recipes, supply every trial's completed calibration path
and use the final version-controlled YAML. The helper permits at most three
global recipes at the feasible extremes and requires ONE final recipe at both calibration
budgets (once when they coincide). The numerical defaults are starting values; no convergence has been
established in the supplied environment.

When CUDA is locally installed and available, the same commands can use
`--device cuda`; keep the pinned versions and scientific recipe. Record the
actual device/runtime. CPU full-matrix training can be time-consuming.

For four feasible budgets, expect 20 definitive run directories. With only
K=1/5 feasible, expect 10, with K=10/20 explicitly marked unavailable. Inspect all
individual results plus `seed_summary.json` and `label_efficiency_seeds.csv`.
Missing seeds fail by default. Do not select the best seed. Existing output
directories fail rather than overwrite. The matrix does not automatically resume.
For an infrastructure failure only, an unchanged missing run may be retried under
a fresh root with the identical frozen receipt and seed. Retain failed attempts.
An optimization-adequacy failure is not fixed by a retry. If any prior run already
exposed test, stop the study: budget/recipe changes require fresh held-out patients.
For one complete summary, copy the verified completed directories into an empty
summary input root with their original `definitive/kXX/seedXXX` paths. Include
exactly one completed result for every prespecified pair, never a best-of-retries
selection. Artifact fingerprints and the receipt are checked by the summarizer.
Once test is released, calibration is
closed on that cohort; changing the recipe requires a new held-out study.

## Recommended Git commit

```text
research: establish an auditable random SegResNet reference

Build a conventional randomly initialized 3D segmentation reference on the
existing patient-level, reader-aware, geometry-preserving LIDC protocol.
Lock nested patient budgets and separate initialization, augmentation, and
sampling randomness. Add patient-balanced reader rotation, step-budgeted
training, train/validation-only calibration, recipe freezing, validation-only
checkpoint selection, pre-test optimization adequacy, and held-out target/QC isolation.

Preserve all seed results with patient-macro overlap and physical metrics,
explicit empty-prediction handling, parameter audits, content fingerprints,
and immutable local provenance. Extend existing CPU CI gates with adversarial
tests, a scientific validator, and synthetic end-to-end execution.

No definitive real LIDC performance, population/support/partition uncertainty,
or causal pretraining benefit is claimed.
```

## Recommended pull request

Title: **Establish an auditable conventional 3D segmentation reference**

```markdown
This adds a conventional random-initialized MONAI SegResNet reference for
oracle-centred localized pulmonary-nodule segmentation, integrated with the
existing patient-level few-shot, cohort-lineage, reader-mask, and physical
preprocessing contracts. It prepares nested 1/5/10/20 patient budgets and five
prespecified initialization seeds without changing partition/support membership.

The reference uses balanced patient/reader sampling, step-based optimization,
train/validation-only calibration, a source/configuration-bound frozen recipe,
validation-only checkpoint locking, pre-test adequacy approval, and final held-out evaluation. Immutable
artifacts preserve every seed, patient-macro physical metrics, empty/undefined
cases, parameter accounting, and reproducibility evidence. Existing CI gates
remain, with explicit compilation, the new scientific validator and synthetic execution added.

Validation: installation/pip check, Ruff, full pytest, Python compilation, all
three scientific validators, synthetic training/evaluation, and wheel/sdist
build passed in the supplied CPU environment. Genuine local LIDC smoke evidence
must be verified before merging; the full repeated-seed matrix remains a later
experiment. No real LIDC performance, screening/diagnosis/deployment claim, or
architecture-controlled causal pretraining benefit is asserted.
```

Before opening the PR, replace the local-gate sentence with the observed
non-sensitive smoke status and counts. Keep the definitive-matrix statement
accurate; remove it only when all feasible K × prespecified-seed runs exist.

Repeated model seeds condition on one fixed patient partition and one fixed nested
support ordering; they do not quantify alternative partitions or support samples.
The test barrier is per-run, not a cohort-wide two-stage release. Real smoke/calibration
keep held-out arrays and target-shape QC closed; oracle centres and eligibility are
explicit label-derived task inputs. Files are integrity-sealed, not encrypted.
