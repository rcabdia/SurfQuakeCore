#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup.py
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
from Cython.Compiler.Errors import CompileError
import numpy as np
import warnings
import platform
import os

system = platform.system().lower()
extra_compile_args = ["/O2"] if system == "windows" else ["-O3"]
extra_libraries = [] if system == "windows" else ["m"]  # link with libm on POSIX
include_dirs = [np.get_include(), "surfquakecore/cython_module"]

def try_cythonize(ext: Extension):
    """Try compiling one extension; warn and skip on error."""
    try:
        return cythonize(
            [ext],
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "nonecheck": False,
            },
        )
    except CompileError as e:
        warnings.warn(f"⚠️ Could not compile {ext.name}: {e}. Skipping.")
    except Exception as e:
        warnings.warn(f"⚠️ Unexpected error compiling {ext.name}: {e}. Skipping.")
    return []

# Explicit list of pyx modules to build
ext_specs = [
    # existing
    ("surfquakecore.cython_module.hampel",       "surfquakecore/cython_module/hampel.pyx"),
    ("surfquakecore.cython_module.whiten",       "surfquakecore/cython_module/whiten.pyx"),

    # CF modules
    ("surfquakecore.cython_module.rec_filter",   "surfquakecore/cython_module/rec_filter.pyx"),
    ("surfquakecore.cython_module.lib_rec_hos",  "surfquakecore/cython_module/lib_rec_hos.pyx"),
    ("surfquakecore.cython_module.lib_rec_rms",  "surfquakecore/cython_module/lib_rec_rms.pyx"),
    ("surfquakecore.cython_module.lib_rosenberger", "surfquakecore/cython_module/lib_rosenberger.pyx"),
    ("surfquakecore.cython_module.lib_rec_cc",   "surfquakecore/cython_module/lib_rec_cc.pyx"),
]

extensions = []
for modname, relsrc in ext_specs:
    if not os.path.isfile(relsrc):
        warnings.warn(f"⚠️ Source not found for {modname}: {relsrc}. Skipping.")
        continue
    ext = Extension(
        name=modname,
        sources=[relsrc],   # relative path!
        include_dirs=include_dirs,
        language="c",
        extra_compile_args=extra_compile_args,
        libraries=extra_libraries,
    )
    extensions.extend(try_cythonize(ext))

class BuildExt(build_ext):
    """Keep hooks open for custom behavior if needed."""
    pass

setup(
    name="surfquake",
    version="0.1.6",
    description="SurfQuake core with Cython accelerations",
    packages=["surfquakecore", "surfquakecore.cython_module"],
    cmdclass={"build_ext": BuildExt},
    ext_modules=extensions,
    include_dirs=include_dirs,
)