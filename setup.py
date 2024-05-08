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
    import glob

    import libucx

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

    # add all the places 'libucx' stores libraries to that paths
    # considered by the linker when compiling extensions
    library_dirs.extend(lib_dirs)

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
