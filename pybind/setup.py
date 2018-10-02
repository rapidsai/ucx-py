# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import absolute_import, print_function

import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


# For demo purposes, we build our own tiny library.
try:
    print("building libmyucp.a")
    print("getcwd: " + str(os.getcwd()))
    assert os.system("gcc -shared -fPIC -c myucp.c -o myucp.o") == 0
    assert os.system("ar rcs libmyucp.a myucp.o") == 0
except:
    if not os.path.exists("libmyucp.a"):
        print("Error building external library, please create libmyucp.a manually.")
        sys.exit(1)

ucx_dir='/home/akvenkatesh/ucx-github/build'
# Here is how to use the library built above.
ext_modules = cythonize([
    Extension("call_myucp",
              sources=["call_myucp.pyx"],
              include_dirs=[os.getcwd(), ucx_dir+'/include', '/cm/extra/apps/CUDA.linux86-64/9.2.88.1_396.26/include'],  # path to .h file(s)
              library_dirs=[os.getcwd(), ucx_dir+'/lib', '/cm/extra/apps/CUDA.linux86-64/9.2.88.1_396.26/lib64'],  # path to .a or .so file(s)
              runtime_library_dirs=[os.getcwd(), ucx_dir+'/lib', '/cm/extra/apps/CUDA.linux86-64/9.2.88.1_396.26/lib64'],
              libraries=['myucp', 'ucp', 'uct', 'ucm', 'ucs', 'cuda', 'cudart'])
])

setup(
    name='Demos',
    ext_modules=ext_modules,
)
