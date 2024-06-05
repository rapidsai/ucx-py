# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import glob
import os
from distutils.sysconfig import get_config_var, get_python_inc

# TODO: delete this before merging. Just checking if this has to be available
#       when setup.py is run.
import libucx  # noqa: F401
from Cython.Distutils.build_ext import new_build_ext
from setuptools import setup
from setuptools.extension import Extension


def _find_libucx_libs_and_headers():
    """
    If the 'libucx' wheel is not installed, returns a tuple of empty lists.
    In that case, the project will be compiled against system installations
    of the UCX libraries.

    If 'libucx' is installed, returns lists of library and header paths to help
    the compiler and linker find its contents. In that case, the project will
    be compiled against those libucx-wheel-provided versions of the UCX libraries.
    """
    try:
        import libucx  # noqa: F811
    except ImportError:
        return [], []

    # find 'libucx'
    module_dir = os.path.dirname(libucx.__file__)

    # find where it stores files like 'libucm.so.0'
    libs = glob.glob(f"{module_dir}/**/lib*.so*", recursive=True)

    # deduplicate those library paths
    lib_dirs = {os.path.dirname(f) for f in libs}
    if not lib_dirs:
        raise RuntimeError(
            f"Did not find shared libraries in 'libucx' install location ({module_dir})"
        )

    # find where it stores headers
    headers = glob.glob(f"{module_dir}/**/include", recursive=True)

    # deduplicate those header paths (and ensure the list only includes directories)
    header_dirs = {f for f in headers if os.path.isdir(f)}
    if not header_dirs:
        raise RuntimeError(
            f"Did not find UCX headers 'libucx' install location ({module_dir})"
        )

    return list(lib_dirs), list(header_dirs)


include_dirs = [os.path.dirname(get_python_inc())]
library_dirs = [get_config_var("LIBDIR")]
libraries = ["ucp", "uct", "ucm", "ucs"]
extra_compile_args = ["-std=c99", "-Werror"]

# tell the compiler and linker where to find UCX libraries and their headers
# provided by the 'libucx' wheel
libucx_lib_dirs, libucx_header_dirs = _find_libucx_libs_and_headers()
library_dirs.extend(libucx_lib_dirs)
include_dirs.extend(libucx_header_dirs)


ext_modules = [
    Extension(
        "ucp._libs.ucx_api",
        sources=["ucp/_libs/ucx_api.pyx", "ucp/_libs/src/c_util.c"],
        depends=["ucp/_libs/src/c_util.h", "ucp/_libs/ucx_api_dep.pxd"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "ucp._libs.arr",
        sources=["ucp/_libs/arr.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": new_build_ext},
    package_data={"ucp": ["VERSION"]},
)
