#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_DATE_STRING})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

bash ci/release/apply_wheel_modifications.sh ${version_override} "-${RAPIDS_PY_CUDA_SUFFIX}"
echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

# Auditwheel rewrites dynamic libraries that are referenced at link time in the
# package. However, UCX loads a number of sub-libraries at runtime via dlopen;
# these are not picked up by auditwheel. Since we have a priori knowledge of
# what these libraries are, we mimic the behaviour of auditwheel by using the
# same hash-based uniqueness scheme and rewriting the link paths.

WHL=$(realpath final_dist/ucx_py*manylinux*.whl)

# first grab the auditwheel hashes for libuc{tms}
LIBUCM=$(unzip -l $WHL | awk 'match($4, /libucm-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCT=$(unzip -l $WHL | awk 'match($4, /libuct-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCS=$(unzip -l $WHL | awk 'match($4, /libucs-[^\.]+\./) { print substr($4, RSTART) }')
LIBNUMA=$(unzip -l $WHL | awk 'match($4, /libnuma-[^\.]+\./) { print substr($4, RSTART) }')

# Extract the libraries that have already been patched in by auditwheel
mkdir -p repair_dist/ucx_py_${RAPIDS_PY_CUDA_SUFFIX}.libs/ucx
unzip $WHL "ucx_py_${RAPIDS_PY_CUDA_SUFFIX}.libs/*.so*" -d repair_dist/

# Patch the RPATH to include ORIGIN for each library
pushd repair_dist/ucx_py_${RAPIDS_PY_CUDA_SUFFIX}.libs
for f in libu*.so*
do
    if [[ -f $f ]]; then
        patchelf --add-rpath '$ORIGIN' $f
    fi
done

popd

# Now copy in all the extra libraries that are only ever loaded at runtime
pushd repair_dist/ucx_py_${RAPIDS_PY_CUDA_SUFFIX}.libs/ucx
cp -P /usr/lib/ucx/* .

# we link against <python>/lib/site-packages/ucx_py_${RAPIDS_PY_CUDA_SUFFIX}.lib/libuc{ptsm}
# we also amend the rpath to search one directory above to *find* libuc{tsm}
for f in libu*.so*
do
    # Avoid patching symlinks, which is redundant
    if [[ ! -L $f ]]; then
        patchelf --replace-needed libuct.so.0 $LIBUCT $f
        patchelf --replace-needed libucs.so.0 $LIBUCS $f
        patchelf --replace-needed libucm.so.0 $LIBUCM $f
        patchelf --replace-needed libnuma.so.1 $LIBNUMA $f
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
zip -r $WHL ucx_py_${RAPIDS_PY_CUDA_SUFFIX}.libs/
popd

RAPIDS_PY_WHEEL_NAME="ucx_py_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
