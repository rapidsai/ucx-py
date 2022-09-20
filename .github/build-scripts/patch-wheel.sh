#!/usr/bin
set -e
set -x

WHL=ucx_py-0.28.0a0+37.ge1f0547-cp38-cp38-manylinux_2_31_x86_64.whl

# first grab the auditwheel hashes for libuc{tms}
LIBUCM=`unzip -l $WHL | grep -E "libucm-.*.so\.0\.0\.0" | awk '{ printf "%s\n",$4 }' | cut -d '-' -f 2 | cut -d '.' -f 1`
LIBUCT=`unzip -l $WHL | grep -E "libuct-.*.so\.0\.0\.0" | awk '{ printf "%s\n",$4 }' | cut -d '-' -f 2 | cut -d '.' -f 1`
LIBUCS=`unzip -l $WHL | grep -E "libucs-.*.so\.0\.0\.0" | awk '{ printf "%s\n",$4 }' | cut -d '-' -f 2 | cut -d '.' -f 1`

mkdir -p ucx_py.libs/ucx
cd ucx_py.libs/ucx
cp -P /usr/lib/ucx/* .

# we link against <python>/lib/site-packages/ucx_py.lib/libuc{ptsm}
# we also amend the rpath to search one directory above to *find* libuc{tsm}
for f in libu*.so.0.0.0
do
  patchelf --replace-needed libuct.so.0 libuct-$LIBUCT.so.0.0.0 $f
  patchelf --replace-needed libucs.so.0 libucs-$LIBUCS.so.0.0.0 $f
  patchelf --replace-needed libucm.so.0 libucm-$LIBUCM.so.0.0.0 $f
  patchelf --add-rpath "\$ORIGIN/.." $f
done
cd -

zip -r $WHL ucx_py.libs/
# python3 -m ucp.benchmarks.send_recv -o numpy -n 100000000 -d 0 -e 1 --reuse-alloc   --backend ucp-core -o cupy
