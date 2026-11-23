# Parameter-efficient adaptation of localized CT features

Can a small task-specific transformation of pretrained CT features improve a pulmonary-nodule probe under limited patient labels? The measured task is reader-specific segmentation in a reference-centered crop. The source model is the [pinned SuPreM SegResNet 2,100-CT release](../configs/foundation/suprem_segresnet.yaml).

The intervention applies a bias-free pointwise residual to the 16-channel, stride-one encoder output. Two 1×1×1 convolutions compute `f + up(ReLU(down(f)))` with bottleneck width four. Zero-initializing the up projection reproduces the control function before training. The original four-scale linear probe supplies 241 parameters and the adapter adds 128. This gives 369 trainable out of 4,399,713 total (0.008387%). The encoder’s 4,399,344 parameters remain frozen. This is output-feature adaptation.

Each experiment crosses pretrained or fixed random features with head-only or head-plus-adapter training. A source pair shares the initial probe, encoder state, locked candidate index, nested patient support, reader rotation, loss, threshold and validation selection. Adaptation and optimization adequacy are scored separately. A blocked adequacy result withholds the within-source Dice difference.

| Data and phase | K | Selected pretrained head Dice | Selected pretrained adapter Dice | Optimization adequacy |
|---|---:|---:|---:|---|
| Synthetic spheres, 12-step smoke | 1 | 0 | 0 | Both blocked |
| Public LIDC-IDRI CT, 12-step smoke | 1 | 0 | 0 | Both blocked |
| Same locked LIDC-IDRI CT, 12-step smoke | 5 | 0 | 0 | Both blocked |
| Same phantom index, secondary calibration | 1 | 0.987705 | 0.993878 | Both passed |

All reported Dice values are patient-macro **selection-validation** values. The secondary phantom comparison gave an adapter-minus-head difference of +0.006173. The longer recipe changed the phase seed, batch size, validation schedule and step budget together. It required 2,800 optimizer steps for the pretrained head and 2,600 for the adapter, including post-selection updates before stopping. The phantom generator derives intensity from each reader’s synthetic target. This limits the inference to optimization of that artificial task.

The convenience pilot comprised ten public single-CT-series Siemens patients and 52 eligible reader targets. Selection used catalog metadata before inspecting annotations. Clinical preprocessing reconciled all ten series. The patient split was six train, two validation and two held out. Four reader targets contributed to validation. In the original 12-step pilot, all four real-data arms had zero selected overlap at both K. At K=5, eleven of 22 eligible support targets were unvisited in 12 updates. In real validation, the pretrained outputs contained foreground but missed every reference voxel. These measurements give a negative short-budget result. Hospital identity, patient independence from pretraining and clinical transfer remain unestablished. Source preparation uses the enrolled scans and annotations. Training, selection and verification opened zero held-out arrays. [TCIA collection and dataset credit](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX) · [Collection design](https://doi.org/10.1118/1.3528204)

On the measured two-thread Linux CPU, the 12-step real pretrained adapter’s sampled peak RSS was approximately 746 MiB at K=1 and 742 MiB at K=5. Corresponding head-only peaks were 709 MiB and 702 MiB. Fresh processes measured RSS at 10-ms intervals. The receipts include initial, final and peak RSS, training and validation time, verified checkpoint-loading time, optimizer membership, gradient/update norms, activation summaries, selection history and SHA-256 identities.

## Fixed 256-update real-CT follow-up

The follow-up retained the patient split, original four arms, support sets, reader rotation, preprocessing, AdamW, head seed 991, effective batch size one and threshold 0.5. It increased exposure and moved validation from every four updates to every 32. These changes jointly affect the available selection checkpoints. The selection rule remained earliest highest patient-macro validation Dice. All arms completed 256 updates. The ordered support-draw hashes agree across arms at each of the eight validation checkpoints. At K=1 each of four targets was visited 64 times. At K=5 all 22 targets were visited 6–26 times under patient-balanced sampling. [Locked follow-up design](../evidence/feature_adaptation/followup/design.json)

| K labeled patients | Pretrained head | Pretrained adapter | Random head | Random adapter |
|---:|---:|---:|---:|---:|
| 1 | 0 at step 32 | 0.022858 at 256 | 0 at 32 | 0.000116 at 256 |
| 5 | 0 at step 32 | 0.015964 at 256 | 0 at 32 | 0 at 32 |

These are descriptive selected validation scores on the same two patients and four reader targets. Both pretrained heads selected a zero-overlap checkpoint. Both pretrained adapters reached the step cap while validation was still improving under the declared patience rule. Only the K=1 random adapter passed its individual engineering adequacy gate. All four within-source paired differences are withheld. A positive measured overlap is compatible with an inconclusive comparison. Source transfer, hospital effects and clinical usefulness remain unestimated. [K=1 full comparison](../evidence/feature_adaptation/followup/ct_k1.json) · [K=5 full comparison](../evidence/feature_adaptation/followup/ct_k5.json)

The same-target diagnostic reused one locked K=1 reader mask. Both functions had zero Dice after 12 presentations and exceeded the declared training Dice of 0.80 by step 128. After 512 presentations the head reached 0.890474 and the adapter 0.900885. Both matched the uncached full model bitwise at the first and final forward checks. For this same training crop, direct physical-coordinate replay from DICOM and 23 XML contours reproduced every mask voxel and all 274,625 normalized CT values to within 1.4×10⁻⁷. This supports learnability and geometric fidelity for that example. [Head memorization](../evidence/feature_adaptation/followup/overfit_suprem_frozen.json) · [Adapter memorization](../evidence/feature_adaptation/followup/overfit_suprem_feature.json) · [Source geometry replay](../evidence/feature_adaptation/followup/geometry.json)

Reference foreground occupied 742 of 1,098,500 validation voxels across the four crops (0.0675%). The selected adapter attained support Dice of 0.86965 and 0.47963 at K=1 and K=5. It overlapped 20 and 14 of the 742 validation reference voxels, respectively, at threshold 0.5. At the original short checkpoints the pretrained arms assigned lower mean probability to validation foreground than background. The descriptive threshold sweep retained 0.5 as the sole primary threshold. The receipts document severe imbalance, poor validation localization and checkpoint selection. Their individual contributions to the failure remain unresolved. The selected head checkpoints and adapter checkpoints occurred at different steps. [Short K=1 thresholds](../evidence/feature_adaptation/followup/threshold_short_k1.json) · [Short K=5](../evidence/feature_adaptation/followup/threshold_short_k5.json) · [Extended K=1](../evidence/feature_adaptation/followup/threshold_extended_k1.json) · [Extended K=5](../evidence/feature_adaptation/followup/threshold_extended_k5.json)

Internal functional readback rebuilt all eight selected models, reproduced their validation logits exactly on this CPU, and recomputed five metrics per reader and their patient-macro aggregates. Empty predictions contribute zero Dice and sensitivity, and undefined HD95 remains null with counts. Worker RSS is sampled process memory, GPU memory is unmeasured. The run receipts preserve selected steps, resource measurements and the complete validation trajectories. The primary upstream publication describes model selection on 1,310 CT volumes. We could not establish patient-level independence of the downstream cohort from either upstream training or selection. Manufacturer alone supplies no hospital identity. [K=1 metric readback](../evidence/feature_adaptation/followup/readback_k1.json) · [K=5 metric readback](../evidence/feature_adaptation/followup/readback_k5.json) · [Upstream study](https://proceedings.iclr.cc/paper_files/paper/2024/file/47360925c45a166c96f652589265dee9-Paper-Conference.pdf)

## Reproduce from the repository root

Use Python 3.12 with the pinned CPU PyTorch build and an editable installation. Place the official checkpoint in the ignored checkpoints directory. These commands use fresh output paths.

```bash
python -m medfm_adapt3d.foundation.feature_experiment --checkpoint checkpoints/supervised_suprem_segresnet_2100.pth --output outputs/feature_phantom
python -m medfm_adapt3d.foundation.feature_experiment --checkpoint checkpoints/supervised_suprem_segresnet_2100.pth --index outputs/feature_phantom/candidates/index.json --partitions outputs/feature_phantom/partitions.json --phase calibration --k 1 --output outputs/feature_phantom_calibration
```

For a prepared clinical cohort, pass its private `--index`, its existing `--partitions`, an eligible `--k` and `--phase smoke`. Run from the same repository root. Every worker writes an exclusive local run directory. Keep its patient IDs, observation rows and task state private. The independent functional readback reproduced selected logits and five validation metrics on all four sets of runs.

For the bounded real follow-up, substitute the private index and partition paths and use a new output directory for each K. Run the same command at K=1 and K=5. The single-mark diagnostic accepts `suprem_frozen` and `suprem_feature` as separate arms.

```bash
python -m medfm_adapt3d.foundation.feature_experiment --checkpoint checkpoints/supervised_suprem_segresnet_2100.pth --index PRIVATE/index.json --partitions PRIVATE/partitions.json --ct-exposure --k 1 --output outputs/ct256-k1
python scripts/diagnose_ct_memorization.py --checkpoint checkpoints/supervised_suprem_segresnet_2100.pth --index PRIVATE/index.json --partitions PRIVATE/partitions.json --arm suprem_feature --output outputs/overfit-feature.json
python scripts/validate_feature_evidence.py
```

## Aggregate evidence

- [Short phantom comparison](../evidence/feature_adaptation/phantom_k1.json)
- [LIDC-IDRI K=1](../evidence/feature_adaptation/lidc_k1.json)
- [LIDC-IDRI K=5](../evidence/feature_adaptation/lidc_k5.json)
- [Secondary phantom calibration](../evidence/feature_adaptation/phantom_calibration_k1.json)
- [Acquisition, historical reproduction and internal verification](../evidence/feature_adaptation/verification.json)
- [Original frozen-representation evidence](../evidence/frozen_representation/verification.json)
