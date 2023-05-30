#!/bin/bash
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
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Get RAPIDS version associated w/ ucx-py version
NEXT_RAPIDS_VERSION="$(curl -sL https://version.gpuci.io/ucx-py/${NEXT_SHORT_TAG})"


echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

sed_runner "s/cudf==.*/cudf==${NEXT_RAPIDS_VERSION}/g" dependencies.yaml

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_RAPIDS_VERSION}/g" "${FILE}"
done

sed_runner "s/^version = .*/version = \"${NEXT_FULL_TAG}\"/g" pyproject.toml
sed_runner "s/^__version__ = .*/__version__ = \"${NEXT_FULL_TAG}\"/g" ucp/__init__.py
