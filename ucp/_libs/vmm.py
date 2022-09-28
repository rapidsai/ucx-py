from cuda import cuda

buf_size = 100
device_num = 0
use_vmm = True

CU_VMM_SUPPORTED = (
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
)
CU_VMM_GDR_SUPPORTED = (
    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
)


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    # elif isinstance(error, cudart.cudaError_t):
    #     return cudart.cudaGetErrorName(error)[1]
    # elif isinstance(error, nvrtc.nvrtcResult):
    #     return cudart.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {error}")


def check_cuda_errors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value} ({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def round_up(x, y):
    return int((x - 1) / y + 1) * y


class VMMArray:
    ptr = None
    size = None

    def __init__(self, device_num, size, align=0):
        self.ptr, self.size = check_cuda_errors(
            self._vmm_alloc(device_num, size, align)
        )
        self.shape = (size,)

    def __del__(self):
        self._vmm_free(self.ptr, self.size)

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.shape),
            "typestr": "u1",
            "data": (self.ptr, False),
            "version": 2,
        }

    def _vmm_free(self, dptr, size):
        if dptr is not None:
            print(f"_vmm_free: {dptr}, {hex(int(dptr))}", flush=True)
            status = cuda.cuMemUnmap(dptr, size)
            if status[0] != cuda.CUresult.CUDA_SUCCESS:
                return status

            status = cuda.cuMemAddressFree(dptr, size)
            return status

    def _vmm_alloc(self, device_num, size, align=0):
        cuDevice = check_cuda_errors(cuda.cuDeviceGet(device_num))
        vmm_attr = check_cuda_errors(
            cuda.cuDeviceGetAttribute(
                CU_VMM_SUPPORTED,
                cuDevice,
            )
        )
        vmm_gdr_attr = check_cuda_errors(
            cuda.cuDeviceGetAttribute(
                CU_VMM_GDR_SUPPORTED,
                cuDevice,
            )
        )

        print(f"Device {device_num} VMM Support: {vmm_attr}")
        print(f"Device {device_num} VMM GDR Support: {vmm_gdr_attr}")

        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_num
        prop.allocFlags.gpuDirectRDMACapable = 1

        status, granularity = cuda.cuMemGetAllocationGranularity(
            prop,
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        assert status == cuda.CUresult.CUDA_SUCCESS

        padded_size = round_up(size, granularity)

        status, dptr = cuda.cuMemAddressReserve(
            padded_size, align, cuda.CUdeviceptr(0), 0
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            print("Error on cuMemAddressReserve", flush=True)
            self._vmm_free(dptr, size)
            return None

        print(f"dptr: {hex(int(dptr))}", flush=True)
        status, allocation_handle = cuda.cuMemCreate(padded_size, prop, 0)
        # check_cuda_errors((status, allocation_handle))
        if status != cuda.CUresult.CUDA_SUCCESS:
            print("Error on cuMemCreate", flush=True)
            self._vmm_free(dptr, size)
            return None

        (status,) = cuda.cuMemMap(dptr, padded_size, 0, allocation_handle, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            print("Error on cuMemMap", flush=True)
            self._vmm_free(dptr, size)
            return None

        (status2,) = cuda.cuMemRelease(allocation_handle)
        if status2 != cuda.CUresult.CUDA_SUCCESS:
            status = status2

        print(dptr, padded_size)

        access_descriptors = [cuda.CUmemAccessDesc()]
        access_descriptors[
            0
        ].location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_descriptors[0].location.id = device_num
        access_descriptors[
            0
        ].flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        (status,) = cuda.cuMemSetAccess(
            dptr, padded_size, access_descriptors, len(access_descriptors)
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            print("Error on cuMemSetAccess", flush=True)
            self._vmm_free(dptr, size)
            return None

        print(f"Allocated: {hex(int(dptr))}", flush=True)
        return (status, dptr, padded_size)

    def copy_from_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_from_host: {hex(int(self.ptr))}", flush=True)
        check_cuda_errors(
            cuda.cuMemcpyHtoDAsync(self.ptr, arr.ctypes.data, self.shape[0], stream)
        )

    def copy_to_host(self, arr, stream=cuda.CUstream(0)):
        print(f"copy_to_host: {hex(int(self.ptr))}", flush=True)
        check_cuda_errors(
            cuda.cuMemcpyDtoHAsync(arr.ctypes.data, self.ptr, self.shape[0], stream)
        )
