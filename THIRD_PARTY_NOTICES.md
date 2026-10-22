# Third-Party Notices

This release preserves source code from an existing research workspace. The following license texts were present locally and have been retained. They describe the corresponding upstream portions; copying them does not assign a new blanket license to unrelated research code.

| Component or source family | Preserved license | Evidence in the local source |
| --- | --- | --- |
| Original `dental/Pytorch_UNet` | GNU General Public License, version 3 | The component contained a `LICENSE` file; it is retained at `coarse_stage/LICENSE` and `licenses/Pytorch-UNet-GPL-3.0.txt`. |
| Nested U-Net implementation family | MIT, copyright 2017 Takato Kimura | The local `dental/pytorch_nested_unet/LICENSE` text is retained at `licenses/pytorch-nested-unet-MIT.txt`; related nested-network classes are retained in the model source. |

Refinement and coarse-stage U-Net blocks share implementation patterns and source comments, including upstream commit references. Preserve these comments and license texts when redistributing corresponding code. The supplied source does not establish complete file-by-file provenance for every research modification.

The upstream coarse-stage README described a generic image-segmentation example. Its task-specific performance claims were not copied into this dental project README. The bundled original `MODEL.pth` has also not been selected or redistributed as a dental checkpoint.

Python packages listed in `requirements.txt` and `requirements-optional.txt` are external dependencies; their code is not vendored into this source folder and their own licenses apply to their distributions.

No new repository-wide MIT, Apache, or other license has been assigned to the original research contributions during this cleanup.

The two README figures are rendered from Figures 1 and 2 of Zhong et al., MICCAI 2019 (DOI: 10.1007/978-3-030-32226-7_60), from the paper supplied for this repository. Published chapter copyright: Springer Nature Switzerland AG 2019. Existing figure rights are unchanged. See `docs/PAPER.md`.
