#!/bin/bash

module load cuda
export UCX_PY_HOME=/home/akvenkatesh/ucx-py/pybind
export LD_LIBRARY_PATH=$HOME/tools/install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/tools/install/bin:$PATH
export PYTHONPATH=$HOME/tools/install/python/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=$UCX_PY_HOME/lib/python3.8/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/ucx-github/build/lib:$LD_LIBRARY_PATH
