#!/usr/bin/env bash

set -Eeoxu pipefail

mypyver=$(python --version)
repair_dir="repair-${mypyver// /_}"

mkdir -p "${repair_dir}" && cd "${repair_dir}"

WHL=$1

# first grab the auditwheel hashes for libuc{tms}
LIBUCM=$(unzip -l $WHL | awk 'match($4, /libucm-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCT=$(unzip -l $WHL | awk 'match($4, /libuct-[^\.]+\./) { print substr($4, RSTART) }')
LIBUCS=$(unzip -l $WHL | awk 'match($4, /libucs-[^\.]+\./) { print substr($4, RSTART) }')

# TODO: This directory is currently hardcoded, but it actually needs to take
# another script argument to get the CUDA suffix used for the current build.
ldd /usr/lib/ucx/libuct_cuda.so
mkdir -p ucx_py.libs/ucx
cd ucx_py.libs/ucx
cp -P /usr/lib/ucx/* .

# we link against <python>/lib/site-packages/ucx_py.lib/libuc{ptsm}
# we also amend the rpath to search one directory above to *find* libuc{tsm}
#for f in libu*.so.0.0.0
#do
#  patchelf --replace-needed libuct.so.0 $LIBUCT $f
#  patchelf --replace-needed libucs.so.0 $LIBUCS $f
#  patchelf --replace-needed libucm.so.0 $LIBUCM $f
#  patchelf --add-rpath '$ORIGIN/..' $f
#done

# bring in cudart as well if avoid symbol collision with other
# libraries e.g. cupy

#find /usr/local/cuda/ -name "libcudart*.so*" | xargs cp -P -t .
#src=libcudart.so
#hash=$(sha256sum ${src} | awk '{print substr($1, 0, 8)}')
#target=$(basename $(readlink -f ${src}))
#
#mv ${target} ${target/libcudart/libcudart-${hash}}
#while readlink ${src} > /dev/null; do
#    target=$(readlink ${src})
#    ln -s ${target/libcudart/libcudart-${hash}} ${src/libcudart/libcudart-${hash}}
#    rm -f ${src}
#    src=${target}
#done
#
#to_rewrite=$(ldd libuct_cuda.so | awk '/libcudart/ { print $1 }')
#patchelf --replace-needed ${to_rewrite} libcudart-${hash}.so libuct_cuda.so
#patchelf --add-rpath '$ORIGIN' libuct_cuda.so

cd -

zip -r $WHL ucx_py.libs/
# python3 -m ucp.benchmarks.send_recv -o numpy -n 100000000 -d 0 -e 1 --reuse-alloc   --backend ucp-core -o cupy
