# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
from distutils.sysconfig import get_config_var, get_python_inc

import versioneer
from setuptools import setup
from setuptools.extension import Extension

include_dirs = [os.path.dirname(get_python_inc())]
library_dirs = [get_config_var("LIBDIR")]
libraries = ["ucp", "uct", "ucm", "ucs", "hwloc"]
extra_compile_args = ["-std=c99"]


ext_modules = [
    Extension(
        "ucp._libs.utils",
        sources=["ucp/_libs/utils.pyx", "ucp/_libs/src/c_util.c"],
        depends=["ucp/_libs/src/c_util.h", "ucp/_libs/core_dep.pxd"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "ucp._libs.send_recv",
        sources=["ucp/_libs/send_recv.pyx", "ucp/_libs/src/c_util.c"],
        depends=[
            "ucp/_libs/src/c_util.h",
            "ucp/_libs/core_dep.pxd",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "ucp._libs.core",
        sources=["ucp/_libs/core.pyx", "ucp/_libs/src/c_util.c"],
        depends=[
            "ucp/_libs/src/c_util.h",
            "ucp/_libs/core_dep.pxd",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "ucp._libs.topological_distance",
        sources=[
            "ucp/_libs/topological_distance.pyx",
            "ucp/_libs/src/topological_distance.c",
        ],
        depends=[
            "ucp/_libs/src/topological_distance.h",
            "ucp/_libs/topological_distance_dep.pxd",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="ucxpy",
    packages=["ucp"],
    ext_modules=ext_modules,
    cmdclass=versioneer.get_cmdclass(),
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
