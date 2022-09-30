from cuda import cuda

from dask_cuda.rmm_vmm_pool import checkCudaErrors


class VmmArray:
    def __init__(self, vmm_allocator, size):
        self.vmm_allocator = vmm_allocator

        self.ptr = cuda.CUdeviceptr(self.vmm_allocator.allocate(size))
        self.shape = (size,)

    def __del__(self):
        self.vmm_allocator.deallocate(int(self.ptr), self.shape[0])

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.shape),
            "typestr": "u1",
            "data": (self.ptr, False),
            "version": 2,
        }

    def copy_from_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_from_host: {hex(int(self.ptr))}", flush=True)
        print(f"copy_from_host: {type(arr)}")
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(self.ptr, arr.ctypes.data, self.shape[0], stream)
        )

    def copy_to_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_to_host: {hex(int(self.ptr))}", flush=True)
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(arr.ctypes.data, self.ptr, self.shape[0], stream)
        )


class VmmArraySlice:
    def __init__(self, ptr, size):
        self.ptr = cuda.CUdeviceptr(ptr)
        self.shape = (size,)

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.shape),
            "typestr": "u1",
            "data": (int(self.ptr), False),
            "version": 2,
        }

    def copy_from_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_from_host: {hex(int(self.ptr))}", flush=True)
        print(f"copy_from_host: {type(arr)}")
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(self.ptr, arr.ctypes.data, self.shape[0], stream)
        )

    def copy_to_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_to_host: {hex(int(self.ptr))}", flush=True)
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(arr.ctypes.data, self.ptr, self.shape[0], stream)
        )


class VmmBlockArray:
    def __init__(self, vmm_block_pool_allocator, size):
        self.vmm_allocator = vmm_block_pool_allocator

        self.ptr = cuda.CUdeviceptr(self.vmm_allocator.allocate(size))
        self.blocks = self.vmm_allocator.get_allocation_blocks(int(self.ptr))
        self.shape = (size,)

    def __del__(self):
        try:
            self.vmm_allocator.deallocate(int(self.ptr), self.shape[0])
        except Exception:
            print("EXCEPTION")

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.shape),
            "typestr": "u1",
            "data": (int(self.ptr), False),
            "version": 2,
        }

    def get_blocks(self):
        return list([VmmArraySlice(block[0], block[1]) for block in self.blocks])

    def copy_from_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_from_host: {hex(int(self.ptr))}", flush=True)
        print(f"copy_from_host: {type(arr)}")
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(self.ptr, arr.ctypes.data, self.shape[0], stream)
        )

    def copy_to_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_to_host: {hex(int(self.ptr))}", flush=True)
        checkCudaErrors(
            cuda.cuMemcpyDtoHAsync(arr.ctypes.data, self.ptr, self.shape[0], stream)
        )
