# Pretrained Checkpoint Inventory

The original local workspace retains both stages of trained checkpoint files. Their paths below are relative to the original project root, not this source-release folder. No checkpoint binaries have been copied into this source release.

## Available files

- Stage 1: 8 checkpoints across four fold directories.
- Stage 2: 9 checkpoints across four fold directories, including temporary snapshots.
- All 17 have parameter names consistent with the retained UNet structure. This is a static inspection of serialization metadata; it is not a successful PyTorch load or a benchmark evaluation.
- Full byte sizes and SHA-256 hashes are recorded in `checkpoint-inventory.json`.

| Original relative path | Size (MiB) | Static identification |
| --- | ---: | --- |
| `dental_review/stage1_model/fold1/CP446.pth` | 51.15 | UNet-style |
| `dental_review/stage1_model/fold1/CP546.pth` | 51.15 | UNet-style |
| `dental_review/stage1_model/fold2/CP187.pth` | 51.15 | UNet-style |
| `dental_review/stage1_model/fold2/CP387.pth` | 51.15 | UNet-style |
| `dental_review/stage1_model/fold3/CP432.pth` | 51.15 | UNet-style |
| `dental_review/stage1_model/fold3/CP632.pth` | 51.15 | UNet-style |
| `dental_review/stage1_model/fold4/CP625.pth` | 51.15 | UNet-style |
| `dental_review/stage1_model/fold4/CP825.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold1/001/Bestmodel_1766.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold1/001/model_tmp.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold2/001/Bestmodel_2385.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold2/001/model_tmp.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold3/001/Bestmodel_4602.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold3/001/model_tmp.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold4/001/Bestmodel_1464.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold4/001/Bestmodel_2545.pth` | 51.15 | UNet-style |
| `dental_review/stage2_model/fold4/001/model_tmp.pth` | 51.15 | UNet-style |

The `Bestmodel_*` filenames are retained from the source experiments. Higher epoch numbers alone do not establish better prediction quality. The cleanup has not reconstructed the historical training split membership or compared these models' scores.

The separate original `dental/Pytorch_UNet/MODEL.pth` was excluded from this dental checkpoint inventory. The accompanying upstream README describes a generic segmentation example, and that file has not been verified as a dental landmark checkpoint.

## Download

[Download `dental-pretrained-unet.zip` from Google Drive](https://drive.google.com/file/d/17sb4rC5OnIgBBA6OrXzYg0wT-iOKzhdD/view?usp=sharing). This link was supplied by the repository maintainer and checked to display the expected archive filename without a sign-in requirement.

The archive is approximately 804 MiB and preserves `stage1_model/` and `stage2_model/` below the `dental-pretrained-unet/` top-level folder. It includes all 17 checkpoints, an inventory, and per-file SHA-256 checksums. The expected archive SHA-256 is:

```text
290dfb582f88f779c680c028c5f02e9035383b13d27847b4491adfdd88af3363
```

Verify this checksum after downloading. The Google Drive copy was not downloaded again during repository preparation; the checksum above was computed from the locally validated package.

## Local use

Copy the selected checkpoint into an ignored local directory, for example `checkpoints/pretrained/`. Keep fold identities aligned when pairing the two stages, and use a preprocessing configuration consistent with how each checkpoint was trained.

For a refinement checkpoint, the intended model configuration is:

```python
import torch
from refine_net.model import UNet

# The retained refinement model module requires CUDA.
net = UNet(n_channels=1, n_classes=20).cuda()
weights = torch.load(
    'checkpoints/pretrained/refinement.pth',
    map_location='cuda',
    weights_only=True,
)
net.load_state_dict(weights, strict=True)
net.eval()
```

This loading pattern is supplied as documentation and was not executed in the cleanup environment because PyTorch/CUDA was unavailable. A matching parameter structure does not establish the correct activation, input normalization, crop geometry, or calibration for a particular checkpoint.

The training `--load` flag in `refine_net/train.py` takes a numbered run directory, rather than an arbitrary checkpoint path. To use that resume mechanism, arrange the selected state dictionary as a `Bestmodel_<epoch>.pth` file within the corresponding run directory. It restores model weights only.

## Sharing weights separately

The checkpoint ZIP is hosted separately on Google Drive. Keep the fold identity, training configuration, validation record, and checksums with any redistributed copies. The binaries are excluded from normal source commits by `.gitignore`.
