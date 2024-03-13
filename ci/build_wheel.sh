#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="ucx-py"
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="pyproject.toml"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" ucp/_version.py

# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

sed -r -i "s/cudf==(.*)\"/cudf${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}

if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi


python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

# Auditwheel rewrites dynamic libraries that are referenced at link time in the
# package. However, UCX loads a number of sub-libraries at runtime via dlopen;
# these are not picked up by auditwheel. Since we have a priori knowledge of
# what these libraries are, we mimic the behaviour of auditwheel by using the
# same hash-based uniqueness scheme and rewriting the link paths.

WHL=$(realpath final_dist/${underscore_package_name}*manylinux*.whl)

# first grab the auditwheel hashes for libuc{tms}
LIBUCM=$(unzip -l $WHL | awk 'match($4, /libucm-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCT=$(unzip -l $WHL | awk 'match($4, /libuct-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCS=$(unzip -l $WHL | awk 'match($4, /libucs-[^\.]+\./) { print substr($4, RSTART) }')

# Extract the libraries that have already been patched in by auditwheel
mkdir -p repair_dist/${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}.libs/ucx
unzip $WHL "${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}.libs/*.so*" -d repair_dist/

# Patch the RPATH to include ORIGIN for each library
pushd repair_dist/${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}.libs
for f in libu*.so*
do
    if [[ -f $f ]]; then
        patchelf --add-rpath '$ORIGIN' $f
    fi
done

popd

# Now copy in all the extra libraries that are only ever loaded at runtime
pushd repair_dist/${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}.libs/ucx
if [[ -d /usr/lib64/ucx ]]; then
    cp -P /usr/lib64/ucx/* .
elif [[ -d /usr/lib/ucx ]]; then
    cp -P /usr/lib/ucx/* .
else
    echo "Could not find ucx libraries"
    exit 1
fi

# we link against <python>/lib/site-packages/${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}.lib/libuc{ptsm}
# we also amend the rpath to search one directory above to *find* libuc{tsm}
for f in libu*.so*
do
    # Avoid patching symlinks, which is redundant
    if [[ ! -L $f ]]; then
        patchelf --replace-needed libuct.so.0 $LIBUCT $f
        patchelf --replace-needed libucs.so.0 $LIBUCS $f
        patchelf --replace-needed libucm.so.0 $LIBUCM $f
        patchelf --add-rpath '$ORIGIN/..' $f
    fi
done

# Bring in cudart as well. To avoid symbol collision with other libraries e.g.
# cupy we mimic auditwheel by renaming the libraries to include the hashes of
# their names. Since there will typically be a chain of symlinks
# libcudart.so->libcudart.so.X->libcudart.so.X.Y.Z we need to follow the chain
# and rename all of them.

find /usr/local/cuda/ -name "libcudart*.so*" | xargs cp -P -t .
src=libcudart.so
hash=$(sha256sum ${src} | awk '{print substr($1, 0, 8)}')
target=$(basename $(readlink -f ${src}))

mv ${target} ${target/libcudart/libcudart-${hash}}
while readlink ${src} > /dev/null; do
    target=$(readlink ${src})
    ln -s ${target/libcudart/libcudart-${hash}} ${src/libcudart/libcudart-${hash}}
    rm -f ${src}
    src=${target}
done

to_rewrite=$(ldd libuct_cuda.so | awk '/libcudart/ { print $1 }')
patchelf --replace-needed ${to_rewrite} libcudart-${hash}.so libuct_cuda.so
patchelf --add-rpath '$ORIGIN' libuct_cuda.so

popd

pushd repair_dist
zip -r $WHL ${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}.libs/
popd

RAPIDS_PY_WHEEL_NAME="${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
