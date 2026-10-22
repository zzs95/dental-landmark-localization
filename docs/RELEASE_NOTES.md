# Source Release Notes

## Scope

The main source is the original `dental/refine_net` directory. The companion original `dental/Pytorch_UNet` source is included as `coarse_stage`. These retain 42 original Python files in total. Two new publication/data-preparation utilities are provided in `tools/`.

The larger research workspace also contained other downloaded frameworks, early MXNet experiments, 2020/2021 exploratory branches, slide assets, notebook outputs, raw images, annotations, and trained models. Those items are not copied into this focused source release. They remain in the original workspace.

`dental_review/refine_net` is not used as the primary source. It shares most of the model/data/helper code with the selected source but has different training behavior: its active model is UNet and checkpoint selection uses validation loss. In the selected source, the original active model was DilatedUNet and selection uses training loss. This release changes the selected source's active model to **UNet**, as requested, while retaining its training-loss selection rule.

## Changes

- Promote the requested refinement source to the top-level `refine_net/` directory.
- Use `UNet(n_channels=1, n_classes=20)` in both the refinement training configuration and the retained prediction script.
- Add command-line data, checkpoint, log, crop, and landmark options to the training entry points.
- Make `--help` available before importing the scientific-computing dependencies.
- Replace workstation path literals with relative placeholders in retained legacy scripts. Those placeholders still need local configuration when running legacy helpers.
- Allow the first refinement run to start in an empty checkpoint directory.
- Save the state dictionary correctly on an interrupted refinement run.
- Remove an unused AdaBound import and load optional CRF code lazily.
- Remove the coarse loader's import-time dependency on a specific validation image and derive image size from the current split instead.
- Allow CPU selection in coarse training and restore training mode after the existing evaluation call.
- Add a preparation tool that validates inputs and writes refinement arrays into a new directory without changing source images or annotations.
- Preserve known upstream license texts and add source provenance records.
- Add Git ignore rules and a dependency-free source-publication check.

The retained network definitions, focal-loss implementation, refinement patch sampling, training-loss checkpoint rule, and prediction algorithms are otherwise preserved. The original prediction scripts remain legacy experiment entry points; this cleanup does not supply a fully integrated two-stage prediction command.

## Validation

The validation record for this local release is saved in `VALIDATION.md` after checks complete. It distinguishes source checks and synthetic preprocessing checks from unexecuted GPU experiments.

## Provenance

`source-manifest.json` maps every retained original Python file to its destination and records the original SHA-256 hash. `release-manifest.json` records hashes for the resulting files, excluding itself and the validation record to avoid circular hashes.

The associated MICCAI 2019 paper is now cited using verified metadata. Published results and two attributed figures are included in the README, with source-to-paper differences documented in PAPER.md. No newly measured benchmark result, hosted model URL, or blanket research-code license has been invented.
