#!/bin/bash
set -euo pipefail

export CFLAGS="${CFLAGS} -idirafter/usr/include"
export CPPFLAGS="${CPPFLAGS} -idirafter/usr/include"
export LDFLAGS="${LDFLAGS} -L/usr/lib -fuse-ld=gold"

if [ -z ${GIT_FULL_HASH-} ]; then
    :;
else
    ./autogen.sh
fi

./configure --prefix="$PREFIX" \
    --disable-cma \
    --disable-numa \
    --enable-mt

make install
