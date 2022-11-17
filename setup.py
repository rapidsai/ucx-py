# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
from distutils.sysconfig import get_config_var, get_python_inc

import versioneer
from Cython.Distutils.build_ext import new_build_ext as build_ext
from setuptools import setup
from setuptools.extension import Extension


# Patch versioneer version for wheel builds.
if "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE" in os.environ:
    orig_get_versions = versioneer.get_versions

    version_override = os.environ["RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE"]

    def get_versions():
        data = orig_get_versions()
        data["version"] = version_override
        return data

    versioneer.get_versions = get_versions


include_dirs = [os.path.dirname(get_python_inc())]
library_dirs = [get_config_var("LIBDIR")]
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

cmdclass = dict(build_ext=build_ext)
cmdclass = versioneer.get_cmdclass(cmdclass)

setup(
    # TODO: At present the ucx-py naming scheme is not dynamic and will not
    # support different CUDA major versions if we need to build wheels. It is
    # hardcoded in pyproject.toml, so overriding it here will not work.
    # name="ucx-py" + os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default=""),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    version=versioneer.get_version(),
)
