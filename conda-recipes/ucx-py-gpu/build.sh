export CFLAGS="${CFLAGS} -idirafter/usr/include"
export CPPFLAGS="${CPPFLAGS} -idirafter/usr/include"
export LDFLAGS="${LDFLAGS} -L/usr/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -fuse-ld=gold"
# symlink libcuda.so.1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs"


$PYTHON setup.py \
    build_ext \
    --with-cuda \
    install --single-version-externally-managed --record=record.txt
