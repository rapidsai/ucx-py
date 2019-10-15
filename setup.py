# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
from distutils.util import strtobool

import versioneer
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension

libraries = ["ucp", "uct", "ucm", "ucs"]
extra_compile_args = ["-std=c99"]


class build_ext(_build_ext):
    user_options = [
        ("with-cuda", None, "build the Cuda extension")
    ] + _build_ext.user_options

    with_cuda = strtobool(os.environ.get("UCX_PY_WITH_CUDA", "0"))

    def run(self):
        if self.with_cuda:
            module = ext_modules[0]
            module.libraries.extend(["cuda", "cudart"])
            module.extra_compile_args.append("-DUCX_PY_CUDA")
        _build_ext.run(self)


ext_modules = [
    Extension(
        "ucp._libs.utils",
        sources=["ucp/_libs/utils.pyx", "ucp/_libs/src/c_util.c"],
        depends=["ucp/_libs/src/c_util.h", "ucp/_libs/core_dep.pxd"],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "ucp._libs.send_recv",
        sources=["ucp/_libs/send_recv.pyx", "ucp/_libs/src/c_util.c"],
        depends=["ucp/_libs/src/c_util.h", "ucp/_libs/core_dep.pxd"],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "ucp._libs.core",
        sources=["ucp/_libs/core.pyx", "ucp/_libs/src/c_util.c"],
        depends=["ucp/_libs/src/c_util.h", "ucp/_libs/core_dep.pxd"],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="ucp",
    packages=["ucp"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext, **versioneer.get_cmdclass()},
    version=versioneer.get_version(),
    python_requires=">=3.6",
    description="Python Bindings for the Unified Communication X library (UCX)",
    long_description=open("README.md").read(),
    author="NVIDIA Corporation",
    license="BSD-3-Clause",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware",
        "Topic :: System :: Systems Administration",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    url="https://github.com/rapidsai/ucx-py",
)
