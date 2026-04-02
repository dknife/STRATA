"""CUDA environment auto-detection for pip-installed CUDA packages."""

import os
import sys


def setup_cuda_env():
    if os.environ.get('CUDA_PATH'):
        return
    for sp_dir in sys.path:
        cuda_rt = os.path.join(sp_dir, 'nvidia', 'cuda_runtime')
        if os.path.isdir(cuda_rt):
            os.environ['CUDA_PATH'] = cuda_rt
            return
    for candidate in [
        os.path.join(os.environ.get('ProgramFiles', ''),
                     'NVIDIA GPU Computing Toolkit', 'CUDA'),
        '/usr/local/cuda',
    ]:
        if os.path.isdir(candidate):
            versions = sorted(os.listdir(candidate), reverse=True)
            if versions:
                os.environ['CUDA_PATH'] = os.path.join(candidate, versions[0])
                return


setup_cuda_env()
