from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = cythonize(
    Extension(
        name="surfquakecore.cython_module.hampel",
        sources=["surfquakecore/cython_module/hampel.pyx"],
        include_dirs=[np.get_include()],
    ),
    compiler_directives={"language_level": "3"},
)

setup(
    ext_modules=extensions,
)