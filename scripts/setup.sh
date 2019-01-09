#!/bin/bash
module load cuda
export LD_LIBRARY_PATH=/home/akvenkatesh/tools/install/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akvenkatesh/tools/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/akvenkatesh/ucx-github/build/lib:$LD_LIBRARY_PATH
export PATH=/home/akvenkatesh/tools/install/bin:$PATH
export PYTHONPATH=/home/akvenkatesh/tools/install/lib/python3.7:$PYTHONPATH
export UCX_PY_CUDA_PATH=/cm/extra/apps/CUDA.linux86-64/9.2.88.1_396.26
export UCX_PY_UCX_PATH=/home/akvenkatesh/ucx-github/build
