# Validation Record

Date: October 22, 2026.

## Completed checks

- All 44 Python files compile successfully with Python 3.12.
- Both training entry points and the data-preparation utility display command-line help without loading ML dependencies.
- The active refinement training and retained prediction configurations both instantiate `UNet(n_channels=1, n_classes=20)`.
- Synthetic data preparation passes with a crop offset and with already cropped square images, including RGB-to-grayscale conversion.
- Output image shape, float32 type, `[0,1]` intensity range, 19-point coordinate shape, crop/scale transforms, and sorted sample order are checked.
- Reusing an existing output directory is rejected without changing the existing arrays.
- Out-of-bounds annotations are rejected before creating an output directory.
- SHA-256 hashes of all 42 retained original source files still match the original workspace files.
- The final source folder contains no model weights, raw clinical image datasets, dataset arrays, archives, notebooks, IDE files, or Python caches.
- Checked credential/private-key patterns have no matches in release text files. This is a pattern-based check rather than a guarantee about arbitrary sensitive text.
- The 17 original two-stage checkpoint files were inventoried and hashed without loading their pickle payloads. Their parameter names are consistent with UNet.

## Not executed

The validation environment did not provide PyTorch or CUDA. Consequently, model loading, GPU training, complete two-stage inference, dependency-version compatibility, historical accuracy, and benchmark reproduction are **not verified**.

The preprocessing utility was executed using a bundled Python 3.12 runtime with NumPy and Pillow, using only synthetic images and annotations created outside the publication folder.

The two allowlisted README figures were visually checked against the supplied paper; title, author order, DOI, venue, page range, and the three reported result rows were verified.
