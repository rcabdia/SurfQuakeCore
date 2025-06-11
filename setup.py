from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="surfquakecore.cython_module.hampel",
        sources=["surfquakecore/cython_module/hampel.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="surfquakecore.cython_module.whiten",
        sources=["surfquakecore/cython_module/whiten.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)