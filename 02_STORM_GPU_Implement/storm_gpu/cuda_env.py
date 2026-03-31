"""
CUDA environment auto-detection for pip-installed CUDA packages.

Ensures CUDA_PATH is set before CuPy is imported, so that NVRTC
and other CUDA libraries can be found from pip packages.
"""

import os
import sys


def setup_cuda_env():
    """Auto-detect and configure CUDA_PATH from pip-installed nvidia packages."""
    if os.environ.get('CUDA_PATH'):
        return  # already set

    # Look for nvidia packages in site-packages
    for sp_dir in sys.path:
        cuda_rt = os.path.join(sp_dir, 'nvidia', 'cuda_runtime')
        if os.path.isdir(cuda_rt):
            os.environ['CUDA_PATH'] = cuda_rt
            return

    # Fallback: standard CUDA Toolkit paths
    for candidate in [
        os.path.join(os.environ.get('ProgramFiles', ''), 'NVIDIA GPU Computing Toolkit', 'CUDA'),
        '/usr/local/cuda',
    ]:
        if os.path.isdir(candidate):
            # Find highest version
            versions = sorted(os.listdir(candidate), reverse=True)
            if versions:
                os.environ['CUDA_PATH'] = os.path.join(candidate, versions[0])
                return


# Auto-setup on import
setup_cuda_env()
