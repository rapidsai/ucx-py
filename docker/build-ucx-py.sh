#!/bin/bash
set -ex

CONDA_HOME=${1:-"/opt/conda"}
CONDA_ENV=${2:-"ucx"}

source ${CONDA_HOME}/etc/profile.d/conda.sh
source ${CONDA_HOME}/etc/profile.d/mamba.sh
mamba activate ${CONDA_ENV}

git clone https://github.com/rapidsai/ucx-py.git
pip install -v ucx-py/
