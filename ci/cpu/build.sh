#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
######################################
# ucx-py CPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function gpuci_logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=-4

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package builds (conda deps: ucx-py)
################################################################################

# logger "Build conda pkg for ucx-py"
# source ci/cpu/ucx-py/build_ucx-py.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

# logger "Upload conda pkgs"
# source ci/cpu/upload_anaconda.sh
