#!/bin/bash
set -euo pipefail

export CFLAGS="${CFLAGS} -idirafter/usr/include"
export CPPFLAGS="${CPPFLAGS} -idirafter/usr/include"
export LDFLAGS="${LDFLAGS} -L/usr/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -fuse-ld=gold"
export LD_LIBRARY_PATH="/usr/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs"

if [ -z ${GIT_FULL_HASH-} ]; then
    :;
else
    ./autogen.sh
fi

./configure --prefix="$PREFIX" \
    --disable-cma \
    --disable-numa \
    --enable-mt \
    --with-cuda=/usr/local/cuda


# confirmed that manually adding /stubs
# to the Makefile "fixes" things. Seems
# like setting LDFLAGS in
# src/ucm to manually include /stubs
# is required...

make install
