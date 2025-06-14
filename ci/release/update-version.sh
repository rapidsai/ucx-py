#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

########################
# ucx-py Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is Major.Minor.Patch - no leading 'v' or trailing 'a'
# Example: 0.30.00
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag | grep -xE 'v[0-9\.]+' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG="${NEXT_MAJOR}.${NEXT_MINOR}"

# Get RAPIDS version associated w/ ucx-py version
NEXT_RAPIDS_SHORT_TAG="$(curl -sL "https://version.gpuci.io/ucx-py/${NEXT_SHORT_TAG}")"

# Need to distutils-normalize the versions for some use cases
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
NEXT_FULL_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_FULL_TAG}'))")
NEXT_RAPIDS_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_RAPIDS_SHORT_TAG}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

DEPENDENCIES=(
  cudf
  rapids-dask-dependency
)
UCX_PY_DEPENDENCIES=(
  ucx-py
)
for FILE in dependencies.yaml conda/environments/*.yml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_RAPIDS_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for DEP in "${UCX_PY_DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
done

for DEP in "${DEPENDENCIES[@]}"; do
  sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_RAPIDS_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" pyproject.toml
done

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_RAPIDS_SHORT_TAG}/g" "${FILE}"
done

echo "${NEXT_FULL_TAG_PEP440}" > VERSION
echo "${NEXT_RAPIDS_SHORT_TAG}.00" > RAPIDS_VERSION
