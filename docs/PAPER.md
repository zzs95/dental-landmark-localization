# Paper and Implementation Notes

## Publication

Zhusi Zhong, Jie Li, Zhenxi Zhang, Zhicheng Jiao, and Xinbo Gao. **An Attention-Guided Deep Regression Model for Landmark Detection in Cephalograms.** In *Medical Image Computing and Computer Assisted Intervention - MICCAI 2019*, LNCS 11769, pages 540-548, Springer, 2019.

- Published chapter: https://doi.org/10.1007/978-3-030-32226-7_60
- Author preprint: https://arxiv.org/abs/1906.07549
- Bibliography: [`citation.bib`](../citation.bib)
- GitHub citation metadata: [`CITATION.cff`](../CITATION.cff)

The publication metadata was checked against the publisher's chapter page and the arXiv record. Method details, figure labels, and reported results were checked against the supplied published-paper PDF.

## Figure provenance

| Repository asset | Source | Content |
| --- | --- | --- |
| `assets/framework.png` | Figure 1, printed page 542 | Global/local stages, Attention-Guide, and U-Net architecture |
| `assets/expansive-exploration.png` | Figure 2, printed page 544 | Expanded proposal regions and five inference patches |

The assets are high-resolution renders of the figure regions from the supplied paper. They are not generated experimental results. Figure attribution: Zhong et al., MICCAI 2019; published chapter copyright Springer Nature Switzerland AG 2019. Existing figure rights are unchanged by this repository's code notices. The full publisher PDF is linked rather than redistributed.

## Method-to-code map

| Paper concept | Retained implementation |
| --- | --- |
| Global U-Net and coarse 20-channel heatmaps | `coarse_stage/unet/`, `coarse_stage/train.py` |
| Patch-based local U-Net | `refine_net/model/`, `refine_net/train.py` |
| Gaussian landmark heatmaps and background channel | `coarse_stage/utils/load.py`, `refine_net/utils/data_crop.py` |
| Sampling training patches around annotations | `refine_net/dataset/crop_data.py` |
| Coarse-coordinate Attention-Guide | `predicteCropDataset` and coarse-label loading in the refinement dataset |
| Expansive Exploration | Expanded offsets in `region_crop` and repeated inference in `predicte_labels_full` |
| Combining BCE and focal loss | `refine_net/train.py`, `refine_net/loss/focal_loss.py` |
| Patch placement and heatmap merging | `refine_net/predicte.py` |

Attention-Guide in this method uses coarse landmark locations to select proposal patches. It should not be confused with a separate transformer attention layer or an SCSE network variant. The selected backbone for both stages is UNet.

## Reproduction differences retained in this source release

The source directory contains historical experimental settings. It is associated with the paper, but not a frozen, fully reproduced configuration for every published result.

- The paper describes a 30-pixel local Gaussian distribution; the retained `refine_net/utils/data_crop.py` currently initializes `normal_distribution(90)`.
- The paper's Figure 2 uses 100-pixel training patches, 150-pixel inference patches, and expansion factor 1.8. The legacy prediction script contains 125-pixel and 200-pixel example configurations and defaults to a test dataset with no coarse labels.
- The original prediction helper can fall back to ground-truth patch centers. Independently predicted coarse labels are required for a valid end-to-end evaluation.
- The paper describes overlap averaging, channel normalization, thresholding at 0.5, and taking the mean position of nonzero pixels. The selected legacy prediction path currently sums merged patches and calls maximum-based coordinate extraction.
- The selected original training source chooses checkpoints using training loss. Validation-based selection from the separate review directory has not been substituted.
- The four-fold checkpoint filenames do not provide a complete reconstruction of historical split membership or evaluation settings.

These differences are documented rather than silently changed during repository preparation. No accuracy claim is made for the current packaged checkpoints without an appropriate reproduction experiment.

## Published evaluation protocol

The paper uses 400 cephalograms annotated with 19 landmarks. The challenge split is 150 training, 150 Test 1, and 100 Test 2 images. Challenge evaluation averages the two observers' annotations; the four-fold experiment uses the senior observer's annotations. The README table transcribes the paper's Table 1 and does not report new measurements from this release.
