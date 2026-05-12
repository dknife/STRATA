"""Build brandes_core.dll from C source.

Handles Korean-path issues by compiling in a temp directory.
Uses MSVC Build Tools with vcvarsall environment setup.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import glob

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, 'brandes_core.c')
_DLL = os.path.join(_DIR, 'brandes_core.dll')

# Known MSVC vcvarsall.bat location
_VCVARSALL = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'


def _try_msvc(src, dll_out, tmpdir):
    """Compile with MSVC via vcvarsall + cl in a temp directory."""
    src_tmp = os.path.join(tmpdir, 'brandes_core.c')
    dll_tmp = os.path.join(tmpdir, 'brandes_core.dll')
    shutil.copy2(src, src_tmp)

    # Use cmd /c to run vcvarsall then cl in one shell
    cmd = (
        f'"{_VCVARSALL}" x64 && '
        f'cl /O2 /openmp /LD /nologo '
        f'brandes_core.c /Fe:brandes_core.dll'
    )
    r = subprocess.run(
        ['cmd', '/c', cmd],
        capture_output=True, text=True, cwd=tmpdir)

    if r.returncode == 0 and os.path.exists(dll_tmp):
        shutil.copy2(dll_tmp, dll_out)
        return True, r.stdout + r.stderr
    return False, r.stdout + r.stderr


def _try_gcc(src, dll_out, tmpdir):
    """Compile with GCC in tmpdir."""
    src_tmp = os.path.join(tmpdir, 'brandes_core.c')
    dll_tmp = os.path.join(tmpdir, 'brandes_core.dll')
    shutil.copy2(src, src_tmp)

    r = subprocess.run(
        ['gcc', '-O3', '-fopenmp', '-shared',
         '-o', dll_tmp, src_tmp, '-lgomp'],
        capture_output=True, text=True, cwd=tmpdir)

    if r.returncode == 0 and os.path.exists(dll_tmp):
        shutil.copy2(dll_tmp, dll_out)
        return True, r.stdout + r.stderr
    return False, r.stdout + r.stderr


def build(verbose=True):
    """Build brandes_core.dll.  Returns path on success, None on failure."""
    if os.path.exists(_DLL):
        if verbose:
            print(f"  DLL already exists: {_DLL}")
        return _DLL

    with tempfile.TemporaryDirectory() as tmpdir:
        # Try MSVC
        if os.path.exists(_VCVARSALL):
            ok, log = _try_msvc(_SRC, _DLL, tmpdir)
            if ok:
                if verbose:
                    print(f"  Built with MSVC + OpenMP: {_DLL}")
                return _DLL
            if verbose:
                print(f"  MSVC failed: {log[-200:]}")

        # Try GCC
        ok, log = _try_gcc(_SRC, _DLL, tmpdir)
        if ok:
            if verbose:
                print(f"  Built with GCC: {_DLL}")
            return _DLL

        if verbose:
            print(f"  BUILD FAILED.\n{log[-500:]}")
    return None


if __name__ == '__main__':
    result = build(verbose=True)
    sys.exit(0 if result else 1)
