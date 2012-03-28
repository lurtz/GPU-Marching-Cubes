#ifndef __gpu_mc_test_kernel_h__
#define __gpu_mc_test_kernel_h__

#include "gpu-mc-kernel.h"

// because of a warning of the cuda compiler, that a pointer on the device may
// point to host memory, the cudaPitchedPtr, which were copied during setup
// phase are compared with the ones that are passed as kernel arguments
__device__ __host__ bool operator==(const cudaPitchedPtr& cpp1, const cudaPitchedPtr& cpp2) {
    return cpp1.pitch == cpp2.pitch && cpp1.ptr == cpp2.ptr && cpp1.xsize == cpp2.xsize && cpp1.ysize == cpp2.ysize;
}

__global__ void cmp_pitched_ptr(unsigned int level, cudaPitchedPtr cpptr, bool * success) {
    cudaPitchedPtr cpptr_device = levels[level];
    *success = cpptr == cpptr_device;
}

#endif
