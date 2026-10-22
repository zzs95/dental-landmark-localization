"""Check source syntax and the source-only publication boundary without ML packages."""
from pathlib import Path
import ast
import json
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    '.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dcm', '.dicom',
    '.npy', '.npz', '.csv', '.pth', '.pt', '.ckpt', '.pkl', '.pickle',
    '.h5', '.hdf5', '.onnx', '.rar', '.zip', '.7z', '.ipynb',
}
FORBIDDEN_DIRS = {'data', 'datasets', 'checkpoints', 'runs', 'logs',
                  'stage1_model', 'stage2_model', '.idea', '__pycache__'}
SKIP_DIRS = {'.git', '.venv', 'venv'}
PAPER_FIGURES = {'assets/framework.png', 'assets/expansive-exploration.png'}


def main():
    errors = []
    source_count = 0
    files = []
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRS:
                errors.append(f'Excluded publication directory: {rel.as_posix()}')
            continue
        files.append(path)
        if (path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.endswith(('.nii.gz', '.tar.gz'))) and rel.as_posix() not in PAPER_FIGURES:
            errors.append(f'Excluded publication file: {rel.as_posix()}')
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f'Unexpected large file: {rel.as_posix()}')
        if path.suffix != '.py':
            continue
        source_count += 1
        try:
            text = path.read_text(encoding='utf-8')
            with warnings.catch_warnings():
                warnings.simplefilter('error', SyntaxWarning)
                tree = ast.parse(text, filename=rel.as_posix())
                compile(tree, rel.as_posix(), 'exec')
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if re.match(r'^[a-zA-Z]:[/\\]|^/(?:home|Users)/', node.value):
                        errors.append(f'Machine-specific path: {rel.as_posix()}:{node.lineno}')
        except (SyntaxError, SyntaxWarning, UnicodeError) as exc:
            errors.append(f'{rel.as_posix()}: {exc}')
    print(json.dumps({'status': 'FAIL' if errors else 'PASS',
                      'python_files': source_count, 'publication_files': len(files),
                      'size_bytes': sum(p.stat().st_size for p in files),
                      'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
