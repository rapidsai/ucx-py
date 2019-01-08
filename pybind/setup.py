# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
import sys

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

UCX_DIR = os.environ.get("UCX_PY_UCX_PATH", "/usr/local/ucx/")
CUDA_DIR = os.environ.get("UCX_PY_CUDA_PATH", "/usr/local/cuda")


msg = "The path '{}' does not exist. Set the {} environment variable."

if not os.path.exists(UCX_DIR):
    print(msg.format(UCX_DIR, "UCX_PY_UCX_PATH"), file=sys.stderr)
    sys.exit(1)


if not os.path.exists(CUDA_DIR):
    print(msg.format(CUDA_DIR, "UCX_PY_CUDA_PATH"), file=sys.stderr)
    sys.exit(1)


print("building libmyucp.a")
print("getcwd: " + str(os.getcwd()))
assert os.system("gcc -shared -fPIC -c myucp.c -o myucp.o") == 0
assert os.system("ar rcs libmyucp.a myucp.o") == 0


ext_modules = cythonize([
    Extension("ucp_py",
              sources=["ucp_py.pyx"],
              include_dirs=[os.getcwd(), UCX_DIR + '/include', CUDA_DIR + '/include'],
              library_dirs=[os.getcwd(), UCX_DIR + '/lib', CUDA_DIR + '/lib64'],
              runtime_library_dirs=[os.getcwd(), UCX_DIR + '/lib', CUDA_DIR + '/lib64'],
              libraries=['myucp', 'ucp', 'uct', 'ucm', 'ucs', 'cuda', 'cudart']),
])

setup(
    name='ucx_py',
    ext_modules=ext_modules
)
