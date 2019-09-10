# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
from distutils.util import strtobool

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext as _build_ext

libraries = [
    'ucp', 'uct', 'ucm', 'ucs'
]
extra_compile_args=['-std=c99']

class build_ext(_build_ext):
    user_options = ([
        ('with-cuda', None, 'build the Cuda extension'),
        ('with-prof', None, 'build with profiling'),
    ] + _build_ext.user_options)

    with_cuda = strtobool(
        os.environ.get("UCX_PY_WITH_CUDA", '0')
    )

    with_prof = strtobool(
        os.environ.get("UCX_PY_WITH_PROF", '0')
    )

    def run(self):
        if self.with_cuda:
            module = ext_modules[0]
            module.libraries.extend(['cuda', 'cudart'])
            module.extra_compile_args.append('-DUCX_PY_CUDA')
        if self.with_prof:
            module = ext_modules[0]
            module.extra_compile_args.append('-DUCX_PY_PROF')
        _build_ext.run(self)


ext_modules = cythonize([
    Extension(
        "ucp._libs.ucp_tiny",
        sources=[
            "ucp/_libs/ucp_tiny.pyx",
            "ucp/_libs/src/c_util.c",
        ],
        depends=[
            "ucp/_libs/src/c_util.h",
            "ucp/_libs/ucp_tiny_dep.pxd",
        ],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    ),
])

setup(
    name='ucp',
    packages=['ucp'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': build_ext,
    }
)
