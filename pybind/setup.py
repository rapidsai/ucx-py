# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions

from __future__ import absolute_import, print_function

import os
import sys

from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Build import cythonize

import os
ucx_dir='/home/akvenkatesh/ucx-github/build'
cuda_dir='/cm/extra/apps/CUDA.linux86-64/9.2.88.1_396.26'
if 'UCX_PY_UCX_PATH' in os.environ.keys():
    ucx_dir=os.environ['UCX_PY_UCX_PATH']
if 'UCX_PY_CUDA_PATH' in os.environ.keys():
    cuda_dir=os.environ['UCX_PY_CUDA_PATH']

try:
    print("building libucp_py_ucp_fxns.a")
    print("getcwd: " + str(os.getcwd()))
    assert os.system("gcc -shared -fPIC -c ucp_py_ucp_fxns.c -o ucp_py_ucp_fxns.o") == 0
    assert os.system("gcc -shared -fPIC -c buffer_ops.c -o buffer_ops.o") == 0
    assert os.system("ar rcs libucp_py_ucp_fxns.a ucp_py_ucp_fxns.o buffer_ops.o") == 0
except:
    if not os.path.exists("libucp_py_ucp_fxns.a"):
        print("Error building external library, please create libucp_py_ucp_fxns.a manually.")
        sys.exit(1)

ext_modules = cythonize([
    Extension("ucp_py",
              sources=["ucp_py.pyx"],
              include_dirs=[os.getcwd(), ucx_dir+'/include', cuda_dir+'/include'],
              library_dirs=[os.getcwd(), ucx_dir+'/lib', cuda_dir+'/lib64'],
              runtime_library_dirs=[os.getcwd(), ucx_dir+'/lib', cuda_dir+'/lib64'],
              libraries=['ucp_py_ucp_fxns', 'ucp', 'uct', 'ucm', 'ucs', 'cuda', 'cudart'])
])

setup(
    name='ucx_py',
    ext_modules=ext_modules
)
