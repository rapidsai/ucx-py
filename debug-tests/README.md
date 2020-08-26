## Debug Tests

Files in this directory are useful for debugging purposes and often require being executed in two separate sessions (tmux/ssh/etc).

NOTE: This was moved outside of the tests directory to prevent users running potentially unstable tests by accident.


## Send/Recv

`send.py` and `recv.py` are used to debug/confirm nvlink message passing over 1000 iterations of either CuPy or cudf objects:

### Process 1

> UCXPY_IFNAME=enp1s0f0 CUDA_VISIBLE_DEVICES=0,1 UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,cuda_copy,cuda_ipc,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm /usr/local/cuda/bin/nvprof python tests/debug-testssend.py

### Process 2

> UCXPY_LOG_LEVEL=DEBUG UCX_LOG_LEVEL=DEBUG UCXPY_IFNAME=enp1s0f0 CUDA_VISIBLE_DEVICES=0,1 UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,cuda_copy,cuda_ipc,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm /usr/local/cuda/bin/nvprof python tests/recv.py

`nvprof` is used to verify NVLINK usage and we are looking at two things primarily:
- existence of [CUDA memcpy PtoP]
- balanced cudaMalloc/cudaFree

### Multi-worker Setup
This setup is particularly useful for IB testing when `multi-node-workers.sh`
is placed in a NFS mount and can be executed independently on each machine

- bash scheduler.sh
- bash multi-node-workers.sh
