# Data Format and Coordinate Conventions

## Refinement arrays

Each split directory must contain exactly one file matching `img_*` and one file matching `label_*`. The legacy loader selects the first match, so do not place multiple array versions in one split directory.

```text
data/scale0.5/
├── TrainData/
│   ├── img_scaled.npy
│   ├── label_scaled.npy
│   ├── sample_ids.txt
│   └── metadata.json
└── ValData/
    ├── img_scaled.npy
    ├── label_scaled.npy
    ├── sample_ids.txt
    └── metadata.json
```

| Array | Shape | Meaning |
| --- | --- | --- |
| `img_scaled.npy` | `(N, S, S)` | Square grayscale images, float32, intensities in `[0,1]` |
| `label_scaled.npy` | `(N, 19, 2)` | Landmarks in image pixels, stored as `(x,y)` |

Images and annotations must share sample order. The preparation tool sorts input filenames lexicographically and records their stems in `sample_ids.txt`. Use zero-padded identifiers when numeric ordering matters.

The training dataset generates 20 heatmap channels: 19 landmarks and a background channel. The original code loads the arrays onto the GPU and constructs full-image heatmaps before extracting patches, so memory usage depends on full image size as well as batch size.

## Input annotations

For `sample001.bmp`, supply `sample001.txt` in the annotation directory. The first 19 lines must each contain exactly two comma-separated numbers, for example `12.5,34.0`. Additional lines are ignored. All points must be finite and inside the image after any requested crop.

The input annotation set must already be selected or averaged. The tool does not discover observer folders or infer which annotation set should be used.

## Cropping and scaling

For a crop with origin `(left,top)` and scale `s`:

```text
x_scaled = (x_original - left) * s
y_scaled = (y_original - top) * s

x_original = x_scaled / s + left
y_original = y_scaled / s + top
```

The historical crop is a 1935 by 1935 square starting at `(0,465)`. With `s=0.5`, the inverse transform becomes `(x,y) * 2 + (0,465)`, which appears in the legacy prediction script. Use that inverse only when those preprocessing settings actually apply.

The preparation tool follows the historical resize rule: truncate the scaled width and round up to an even value. It multiplies coordinates by the requested scale. For a 1935-pixel crop and scale `0.5`, the output width is 968; the small difference between nominal and effective image scale is retained from the research preprocessing and should be accounted for in precision-sensitive reproduction work.

## Coarse-stage training layout

The coarse training script expects already cropped square images and annotations in the cropped image coordinate frame:

```text
data/Cephalometry_crop/
├── RawImage/
│   ├── TrainData/              # .bmp images
│   ├── TestData/               # .bmp images
│   └── ValData/                # .bmp images
└── AnnotationsByMD/
    └── 400_ave/                # matching <image-stem>.txt annotations
```

The `400_ave` directory name is retained from the original code. It does not cause annotation averaging. The coarse script uses `TestData` in its existing evaluation loop; establish an appropriate held-out evaluation protocol for your own experiments rather than inferring split roles from these legacy names.

## Coarse predictions for refinement

The original prediction dataset loads a coarse-label array shaped `(N,19,2)`. Its helper splits a concatenated array into rows `0:150`, `150:300`, and `300:400`. These boundaries are historical assumptions, not a general data-splitting mechanism.

The coarse coordinates must use the same scaled/cropped pixel frame as the refinement image arrays. Configure both the preload path and split mapping in the legacy prediction code when using a different split. Missing coarse predictions must not be silently replaced by ground-truth centers when reporting end-to-end results.
