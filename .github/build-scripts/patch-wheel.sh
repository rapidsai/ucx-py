#!/usr/bin/env bash
# Auditwheel rewrites dynamic libraries that are referenced at link time in the
# package. However, UCX loads a number of sub-libraries at runtime via dlopen;
# these are not picked up by auditwheel. Since we have a priori knowledge of
# what these libraries are, we mimic the behaviour of auditwheel by using the
# same hash-based uniqueness scheme and rewriting the link paths.

set -Eeoxu pipefail

mypyver=$(python --version)
repair_dir="repair-${mypyver// /_}"

mkdir -p "${repair_dir}" && cd "${repair_dir}"

WHL=$1

# first grab the auditwheel hashes for libuc{tms}
LIBUCM=$(unzip -l $WHL | awk 'match($4, /libucm-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCT=$(unzip -l $WHL | awk 'match($4, /libuct-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCS=$(unzip -l $WHL | awk 'match($4, /libucs-[^\.]+\./) { print substr($4, RSTART) }')
LIBNUMA=$(unzip -l $WHL | awk 'match($4, /libnuma-[^\.]+\./) { print substr($4, RSTART) }')

# TODO: This directory is currently hardcoded, but it actually needs to take
# another script argument to get the CUDA suffix used for the current build.
mkdir -p ucx_py_cu11.libs/ucx
cd ucx_py_cu11.libs/ucx
cp -P /usr/lib/ucx/* .

# we link against <python>/lib/site-packages/ucx_py_cu11.lib/libuc{ptsm}
# we also amend the rpath to search one directory above to *find* libuc{tsm}
for f in libu*.so*
do
  patchelf --replace-needed libuct.so.0 $LIBUCT $f
  patchelf --replace-needed libucs.so.0 $LIBUCS $f
  patchelf --replace-needed libucm.so.0 $LIBUCM $f
  patchelf --replace-needed libnuma.so.1 $LIBNUMA $f
  patchelf --add-rpath '$ORIGIN/..' $f
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

cd -

zip -r $WHL ucx_py_cu11.libs/
