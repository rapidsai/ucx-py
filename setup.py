# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
from distutils.sysconfig import get_config_var, get_python_inc

from Cython.Distutils.build_ext import new_build_ext
from setuptools import setup
from setuptools.extension import Extension

library_dirs = [get_config_var("LIBDIR")]

# if the 'libucx' wheel was installed, add its library location to
# library_dirs so it'll be linked against
#
# NOTE: doing this '.__file__' stuff because `sysconfig.get_path("platlib")` and similar
#       can return paths to the main Python interpreter's installation locations...
#       not where 'libucx' is going to be installed at build time when using
#       build isolation
try:
    import libucx

    libucx_libdir = os.path.join(os.path.dirname(libucx.__file__), "lib")
    library_dirs.append(libucx_libdir)
except ImportError:
    pass

include_dirs = [os.path.dirname(get_python_inc())]
libraries = ["ucp", "uct", "ucm", "ucs"]
extra_compile_args = ["-std=c99", "-Werror"]


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
