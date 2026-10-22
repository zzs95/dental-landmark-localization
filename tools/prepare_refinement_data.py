"""Prepare one refinement split from images and matching averaged landmark text files."""
import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='one image split')
    parser.add_argument('--annotations', type=Path, required=True,
                        help='matching <image-stem>.txt files, first 19 lines x,y')
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--crop-box', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                        help='optional PIL crop box in original image pixels')
    return parser.parse_args()


def main():
    args = get_args()
    if not 0 < args.scale <= 1:
        raise SystemExit('--scale must be in (0, 1].')
    if not args.images.is_dir() or not args.annotations.is_dir():
        raise SystemExit('Both input directories must exist.')
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new directory to avoid overwriting data.')
    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'})
    if not paths:
        raise SystemExit('No supported images found.')
    if len({p.stem for p in paths}) != len(paths):
        raise SystemExit('Image stems must be unique.')
    import numpy as np
    from PIL import Image

    crop = args.crop_box
    if crop and (crop[0] < 0 or crop[1] < 0 or crop[2] <= crop[0] or crop[3] <= crop[1]):
        raise SystemExit('Invalid crop box.')
    all_labels = []
    expected_size = None
    for path in paths:
        with Image.open(path) as image:
            if crop and (crop[2] > image.width or crop[3] > image.height):
                raise SystemExit(f'Crop box exceeds image bounds: {path.name}')
            size = (crop[2] - crop[0], crop[3] - crop[1]) if crop else image.size
            if size[0] != size[1] or (expected_size is not None and size != expected_size):
                raise SystemExit('Prepared images must be square and all have the same size.')
            expected_size = size
        lines = (args.annotations / f'{path.stem}.txt').read_text(encoding='utf-8-sig').splitlines()[:19]
        labels = np.asarray([[float(v) for v in line.split(',')] for line in lines], dtype=np.float32)
        if labels.shape != (19, 2) or not np.isfinite(labels).all():
            raise SystemExit(f'Expected 19 finite x,y coordinates: {path.stem}.txt')
        if crop:
            labels -= np.asarray(crop[:2], dtype=np.float32)
        if (labels < 0).any() or (labels[:, 0] >= size[0]).any() or (labels[:, 1] >= size[1]).any():
            raise SystemExit(f'Landmarks outside prepared image: {path.stem}.txt')
        all_labels.append(labels * args.scale)

    scaled_width = int(expected_size[0] * args.scale)
    scaled_width += scaled_width % 2
    if scaled_width < 2:
        raise SystemExit('Scaled image is too small.')
    args.output.mkdir(parents=True)
    images = np.lib.format.open_memmap(args.output / 'img_scaled.npy', mode='w+', dtype=np.float32,
                                     shape=(len(paths), scaled_width, scaled_width))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert('L')
            if crop:
                image = image.crop(tuple(crop))
            image = image.resize((scaled_width, scaled_width), resample=Image.Resampling.BICUBIC)
            images[index] = np.asarray(image, dtype=np.float32) / 255.0
    images.flush()
    del images
    np.save(args.output / 'label_scaled.npy', np.stack(all_labels))
    (args.output / 'sample_ids.txt').write_text('\n'.join(p.stem for p in paths) + '\n', encoding='utf-8')
    (args.output / 'metadata.json').write_text(json.dumps({
        'samples': len(paths), 'landmarks': 19, 'scale': args.scale,
        'crop_box': crop, 'image_width': scaled_width,
        'coordinate_order': 'x,y', 'intensity_range': [0, 1],
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(paths)} images in {args.output}')


if __name__ == '__main__':
    main()
