"""Strict, Visible Assumptions for the Conventional Reference Experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

Phase = Literal["smoke", "calibration", "definitive"]
TASK = "oracle-centred localized 3D pulmonary-nodule segmentation"
METRICS = ("dice", "hd95_mm", "absolute_volume_error_mm3", "voxel_sensitivity")
LIMITS = (
    "Reference-annotation centres are supplied; this is localized segmentation.",
    "No whole-scan detection, autonomous localization, screening, or diagnosis.",
    "No clinical deployment or clinically sufficient K claim.",
    "No cancer outcome prediction or population generalizability claim.",
    "Different architectures do not isolate the causal effect of pretraining.",
    "Seed spread is stochastic variability under a fixed cohort, not population uncertainty.",
    "Repeated model seeds condition on one fixed patient partition and nested support ordering; "
    "they do not quantify alternative partitions or alternative K-patient support samples.",
    "Oracle centres and preparation eligibility are label-derived task inputs; additional "
    "held-out target content/QC is restricted to authorized target access.",
)


@dataclass(frozen=True, slots=True)
class DataConfig:
    budgets: tuple[int, ...]
    unit: str
    nested_support: bool
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True, slots=True)
class Seeds:
    partition: int
    support: int
    model: tuple[int, ...]
    augmentation: int
    loader: int
    calibration: int
    smoke: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str
    initialization: str
    init_filters: int
    blocks_down: tuple[int, ...]
    blocks_up: tuple[int, ...]
    norm_groups: int
    upsample_mode: str


@dataclass(frozen=True, slots=True)
class Optimization:
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    loss: str
    dice_weight: float
    ce_weight: float
    gradient_clip_norm: float
    max_steps: int
    min_steps: int
    validation_interval: int
    patience: int
    min_delta: float


@dataclass(frozen=True, slots=True)
class SmokeBudget:
    max_steps: int
    min_steps: int
    validation_interval: int
    patience: int


@dataclass(frozen=True, slots=True)
class Sampling:
    strategy: str
    flip_probability: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    threshold: float
    selection_metric: str
    selection_tie_break: str
    test_use: str
    hd95_undefined_policy: str
    metrics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Reporting:
    seed_statistics: tuple[str, ...]
    seed_uncertainty: str
    population_inference: bool


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    schema_version: str
    task: str
    data: DataConfig
    seeds: Seeds
    model: ModelConfig
    optimization: Optimization
    smoke: SmokeBudget
    sampling: Sampling
    evaluation: Evaluation
    reporting: Reporting

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or self.task != TASK:
            raise ValueError("Unsupported baseline schema or task definition.")
        if self.data.budgets != (1, 5, 10, 20):
            raise ValueError("The reference contract requires patient budgets (1, 5, 10, 20).")
        if self.data.unit != "patient" or self.data.nested_support is not True:
            raise ValueError("Patient units and nested support cannot be disabled.")
        fractions = (self.data.validation_fraction, self.data.test_fraction)
        if not all(math.isfinite(f) and 0 < f < 1 for f in fractions) or sum(fractions) >= 1:
            raise ValueError("Partition fractions must leave a non-empty training pool.")
        if len(self.seeds.model) < 5 or len(set(self.seeds.model)) != len(self.seeds.model):
            raise ValueError("At least five distinct definitive model seeds are required.")
        seed_values = (
            *self.seeds.model,
            self.seeds.partition,
            self.seeds.support,
            self.seeds.augmentation,
            self.seeds.loader,
            self.seeds.calibration,
            self.seeds.smoke,
        )
        if any(type(s) is not int or not 0 <= s < 2**32 for s in seed_values):
            raise ValueError("Seeds must be non-negative 32-bit integers.")
        if {self.seeds.calibration, self.seeds.smoke} & set(self.seeds.model):
            raise ValueError("Engineering seeds must be separate from definitive model seeds.")
        m = self.model
        if (m.architecture, m.initialization, m.upsample_mode) != ("SegResNet", "random", "deconv"):
            raise ValueError("Only random-initialized SegResNet with deconvolution is allowed.")
        if len(m.blocks_down) != 4 or len(m.blocks_up) != 3:
            raise ValueError("This reference uses four encoder and three decoder stages.")
        integers = (m.init_filters, m.norm_groups, *m.blocks_down, *m.blocks_up)
        if any(type(v) is not int or v <= 0 for v in integers):
            raise ValueError("Model widths, blocks, and groups must be positive integers.")
        if m.init_filters % m.norm_groups:
            raise ValueError("init_filters must be divisible by norm_groups.")
        o = self.optimization
        if o.optimizer != "AdamW" or o.loss != "DiceCELoss":
            raise ValueError("The reference optimizer/loss are AdamW and DiceCELoss.")
        positive = (o.learning_rate, o.dice_weight, o.ce_weight, o.gradient_clip_norm)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Optimization rates, loss weights, and clip norm must be positive.")
        if not math.isfinite(o.weight_decay) or o.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative.")
        if type(o.batch_size) is not int or o.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        for budget in (o, self.smoke):
            counts = (
                budget.min_steps,
                budget.max_steps,
                budget.validation_interval,
                budget.patience,
            )
            if any(type(v) is not int or v <= 0 for v in counts):
                raise ValueError("Step budgets and patience must be positive integers.")
            if budget.min_steps > budget.max_steps:
                raise ValueError("min_steps must not exceed max_steps.")
            if budget.validation_interval > budget.max_steps:
                raise ValueError("At least one validation evaluation is required.")
        if not math.isfinite(o.min_delta) or o.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative.")
        if self.sampling.strategy != "patient-balanced-reader-rotation":
            raise ValueError("The reference requires patient-balanced reader rotation.")
        if self.sampling.flip_probability != 0:
            raise ValueError("The conservative LPS reference disables all spatial reflections.")
        e = self.evaluation
        if not math.isfinite(e.threshold) or not 0 < e.threshold < 1:
            raise ValueError("The prespecified probability threshold must lie in (0, 1).")
        if (e.selection_metric, e.selection_tie_break, e.test_use, e.hd95_undefined_policy) != (
            "patient_macro_dice",
            "earliest_step",
            "once_after_optimization_adequacy",
            "null_with_counts",
        ):
            raise ValueError(
                "Validation selection, test isolation, and undefined metrics are locked."
            )
        if e.metrics != (*METRICS, "empty_prediction"):
            raise ValueError("All prespecified complementary metrics are required.")
        r = self.reporting
        if r.seed_statistics != ("mean", "std", "median", "iqr", "min", "max"):
            raise ValueError("All-seed reporting statistics are required.")
        if r.population_inference is not False or r.seed_uncertainty != (
            "stochastic_initialization_under_fixed_cohort"
        ):
            raise ValueError("Initialization spread must not be called population uncertainty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def budget(self, phase: Phase) -> Optimization:
        if phase not in ("smoke", "calibration", "definitive"):
            raise ValueError(f"Unknown experiment phase: {phase}.")
        if phase == "smoke":
            return replace(self.optimization, **asdict(self.smoke), batch_size=1)
        return self.optimization


def load_baseline_config(path: Path) -> BaselineConfig:
    """Reject unknown/missing settings, including accidental pretrained/checkpoint fields."""
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {f.name for f in fields(BaselineConfig)}
    if not isinstance(root, dict) or set(root) != expected:
        raise ValueError(f"Baseline configuration must contain exactly {sorted(expected)}.")
    classes = {
        "data": DataConfig,
        "seeds": Seeds,
        "model": ModelConfig,
        "optimization": Optimization,
        "smoke": SmokeBudget,
        "sampling": Sampling,
        "evaluation": Evaluation,
        "reporting": Reporting,
    }
    for name, cls in classes.items():
        values = root[name]
        required = {f.name for f in fields(cls)}
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError(
                f"{name} must contain exactly {sorted(required)}; unknown keys rejected."
            )
        values = {
            key: tuple(value) if isinstance(value, list) else value for key, value in values.items()
        }
        root[name] = cls(**values)
    return BaselineConfig(**root)
