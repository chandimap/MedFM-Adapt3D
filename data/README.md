# Data directory

Raw medical images are intentionally not version-controlled.

Planned local layout:

```text
data/
├── raw/          # original downloaded data; immutable
├── interim/      # parsed/converted intermediate artefacts
├── processed/    # deterministic model-ready derivatives
├── cache/        # disposable caches
└── manifests/    # small machine-readable manifests that may be versioned