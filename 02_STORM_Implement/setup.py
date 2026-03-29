from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "storm._storm_core",
        ["storm/_storm_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="storm-cython",
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    }),
)
