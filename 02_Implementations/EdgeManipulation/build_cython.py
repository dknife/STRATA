"""
Build Cython extensions for EdgeMan_BFS and EdgeMan_DLevel.

Handles Windows Korean-path issue by building in a temp directory
and copying the .pyd back.

Usage:
    cd 02_Implementations
    python -m EdgeManipulation.build_cython
"""

import os
import sys
import shutil
import tempfile
import subprocess
import sysconfig


def find_vcvarsall():
    """Find vcvarsall.bat for MSVC environment setup."""
    candidates = [
        r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat',
        r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat',
        r'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat',
        r'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat',
        r'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def get_ext_suffix():
    return sysconfig.get_config_var('EXT_SUFFIX') or '.pyd'


def find_numpy_include():
    import numpy
    return numpy.get_include()


def build_module(pyx_path, module_name, output_dir):
    """Build a single .pyx -> .pyd using temp directory to avoid Korean path issues."""
    ext_suffix = get_ext_suffix()
    numpy_inc = find_numpy_include()
    python_inc = sysconfig.get_path('include')
    python_lib_dir = os.path.join(sys.prefix, 'libs')

    # Find python library name
    ver = f'{sys.version_info.major}{sys.version_info.minor}'
    python_lib = f'python{ver}.lib'

    print(f"\n{'='*60}")
    print(f"Building {module_name}")
    print(f"  pyx: {pyx_path}")
    print(f"  output: {output_dir}")
    print(f"  suffix: {ext_suffix}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory(prefix='cython_build_') as tmpdir:
        # 1. Copy .pyx to temp dir
        tmp_pyx = os.path.join(tmpdir, f'{module_name}.pyx')
        shutil.copy2(pyx_path, tmp_pyx)

        # 2. Cython: .pyx -> .c
        print("  [1/3] Cython -> C ...")
        rc = subprocess.run(
            [sys.executable, '-m', 'cython', tmp_pyx],
            cwd=tmpdir, capture_output=True, text=True
        )
        if rc.returncode != 0:
            print(f"  Cython FAILED:\n{rc.stderr}")
            return False

        tmp_c = os.path.join(tmpdir, f'{module_name}.c')
        if not os.path.exists(tmp_c):
            print(f"  Cython produced no .c file")
            return False

        # 3. Try setuptools first (works if no Korean in temp path)
        print("  [2/3] Compile ...")
        setup_py = os.path.join(tmpdir, 'setup.py')
        with open(setup_py, 'w') as f:
            f.write(f"""
import sys
sys.argv = ['setup.py', 'build_ext', '--inplace']
from setuptools import setup, Extension
import numpy
setup(
    ext_modules=[Extension(
        '{module_name}',
        ['{module_name}.c'],
        include_dirs=[numpy.get_include()],
    )],
)
""")
        rc = subprocess.run(
            [sys.executable, setup_py],
            cwd=tmpdir, capture_output=True, text=True
        )

        pyd_name = f'{module_name}{ext_suffix}'
        tmp_pyd = os.path.join(tmpdir, pyd_name)

        if rc.returncode != 0 or not os.path.exists(tmp_pyd):
            # Fallback: manual cl + link via vcvarsall batch script
            print(f"  setuptools failed, trying vcvarsall + cl + link ...")
            print(f"  stderr: {rc.stderr[:200]}")

            vcvarsall = find_vcvarsall()
            if vcvarsall is None:
                print("  vcvarsall.bat not found. Install VS Build Tools.")
                return False

            obj_name = f'{module_name}.obj'

            # Write a .bat that sets up MSVC env and runs cl + link
            bat_path = os.path.join(tmpdir, 'build.bat')
            with open(bat_path, 'w') as bf:
                bf.write(f'@echo off\n')
                bf.write(f'call "{vcvarsall}" amd64 >nul 2>&1\n')
                bf.write(f'cl /c /O2 /MD /nologo ')
                bf.write(f'/I"{python_inc}" /I"{numpy_inc}" ')
                bf.write(f'/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ')
                bf.write(f'{module_name}.c /Fo{obj_name}\n')
                bf.write(f'if errorlevel 1 exit /b 1\n')
                bf.write(f'link /DLL /nologo ')
                bf.write(f'/LIBPATH:"{python_lib_dir}" ')
                bf.write(f'{python_lib} {obj_name} /OUT:{pyd_name}\n')
                bf.write(f'if errorlevel 1 exit /b 2\n')

            rc2 = subprocess.run(
                ['cmd', '/c', bat_path],
                cwd=tmpdir, capture_output=True, text=True
            )
            if rc2.returncode != 0:
                print(f"  Build FAILED (exit {rc2.returncode}):")
                print(f"  stdout: {rc2.stdout[-500:]}")
                print(f"  stderr: {rc2.stderr[-500:]}")
                return False

            if not os.path.exists(tmp_pyd):
                print(f"  .pyd not found after manual build")
                return False

        # 4. Copy .pyd to output dir
        dest_pyd = os.path.join(output_dir, pyd_name)
        shutil.copy2(tmp_pyd, dest_pyd)
        print(f"  [3/3] Copied -> {dest_pyd}")
        print(f"  SUCCESS: {os.path.getsize(dest_pyd)} bytes")
        return True


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    modules = [
        (
            os.path.join(base, 'EdgeMan_BFS', '_core.pyx'),
            '_core',
            os.path.join(base, 'EdgeMan_BFS'),
        ),
        (
            os.path.join(base, 'EdgeMan_DLevel', '_core.pyx'),
            '_core',
            os.path.join(base, 'EdgeMan_DLevel'),
        ),
    ]

    ok = True
    for pyx_path, mod_name, out_dir in modules:
        if not os.path.exists(pyx_path):
            print(f"SKIP: {pyx_path} not found")
            continue
        if not build_module(pyx_path, mod_name, out_dir):
            print(f"FAILED: {pyx_path}")
            ok = False

    if ok:
        print("\nAll modules built successfully.")
    else:
        print("\nSome modules failed. Check errors above.")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
