#!/usr/bin/env python3
"""Verify that built distributions contain runtime data and complete CUDA sources."""

from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive, = (ROOT / 'dist').glob('*.tar.gz')
    wheel, = (ROOT / 'dist').glob('*.whl')
    with tarfile.open(archive) as package:
        names = {name.split('/', 1)[1] for name in package.getnames() if '/' in name}
        required = {'pyproject.toml', 'requirements.txt', 'scripts/install.sh', 'scripts/validate_gpu.py',
                    'models/prompt_tensor/posi_prompt.pth', 'Block-Sparse-Attention/pyproject.toml',
                    'Block-Sparse-Attention/csrc/cutlass/include/cutlass/cutlass.h',
                    'Block-Sparse-Attention/csrc/cutlass/LICENSE.txt'}
        required.update(str(path.relative_to(ROOT)) for path in (ROOT / 'Block-Sparse-Attention/csrc').rglob('*')
                        if path.is_file() and path.suffix in ('.h', '.hpp', '.cuh', '.cpp', '.cu'))
        missing = required - names
        if missing:
            raise RuntimeError(f'Source distribution is incomplete: {sorted(missing)[:10]}')
        if any('/cutlass/docs/' in name or name.endswith('.mp4') for name in names):
            raise RuntimeError('Generated docs or sample videos leaked into the source distribution')
    with zipfile.ZipFile(wheel) as package:
        names = package.namelist()
        if 'utils/runtime.py' not in names or not any(name.startswith('diffsynth/tokenizer_configs/') for name in names):
            raise RuntimeError('Wheel is missing runtime modules or tokenizer data')
        metadata = BytesParser().parsebytes(package.read(next(name for name in names if name.endswith('.dist-info/METADATA'))))
        dependencies = metadata.get_all('Requires-Dist')
        if len([item for item in dependencies if item.startswith('opencv-')]) != 1:
            raise RuntimeError('Wheel metadata must select exactly one OpenCV distribution')
        if metadata['Requires-Python'] != '<3.15,>=3.12':
            raise RuntimeError('Unexpected Python version range in wheel metadata')
    print(f'Distributions verified: {archive.name}, {wheel.name}')


if __name__ == '__main__':
    main()
