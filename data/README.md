# Data directory

Raw medical images and source annotations are intentionally not
version-controlled.

Local layout:

```text
data/
├── raw/          # immutable source data
├── interim/      # deterministic intermediate artefacts
├── processed/    # model-ready derivatives
├── cache/        # disposable caches
└── manifests/    # machine-readable cohort provenance and audits