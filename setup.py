# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
import re
from distutils.sysconfig import get_config_var, get_python_inc

from Cython.Build import cythonize
from Cython.Distutils.build_ext import new_build_ext as build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

import versioneer

with open("README.md", "r") as fh:
    readme = fh.read()

include_dirs = [os.path.dirname(get_python_inc())]
library_dirs = [get_config_var("LIBDIR")]
libraries = ["ucp", "uct", "ucm", "ucs"]
extra_compile_args = ["-std=c99", "-Werror"]


def get_ucp_version():
    with open(include_dirs[0] + "/ucp/api/ucp_version.h") as f:
        ftext = f.read()
        major = re.findall("^#define.*UCP_API_MAJOR.*", ftext, re.MULTILINE)
        minor = re.findall("^#define.*UCP_API_MINOR.*", ftext, re.MULTILINE)

        major = int(major[0].split()[-1])
        minor = int(minor[0].split()[-1])

        return (major, minor)


ext_modules = cythonize(
    [
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
    ],
)

cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext

install_requires = [
    "numpy",
    "psutil",
    "pynvml",
]

extras_require = {
    "test": [
        "pytest",
        "pytest-asyncio",
    ],
}

setup(
    name="ucx-py",
    packages=find_packages(exclude=["tests*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    version=versioneer.get_version(),
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras_require,
    description="Python Bindings for the Unified Communication X library (UCX)",
    long_description=readme,
    author="NVIDIA Corporation",
    license="BSD-3-Clause",
    license_files=["LICENSE"],
    zip_safe=False,
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
    ],
    url="https://github.com/rapidsai/ucx-py",
)
