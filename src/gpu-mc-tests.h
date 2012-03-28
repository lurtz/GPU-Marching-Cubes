/* see LICENSE file for copyright and license details */

#ifndef __gpu_mc_tests__
#define __gpu_mc_tests__

#include "gpu-mc-test-kernel.h"
#include <vector>
#include <utility>

// forward declarations from gpu-mc.cu
// these functions are needed by the tests
void updateScalarField();
template<typename T>
T* get_data_from_pitched_ptr(unsigned int level);
template<typename T>
unsigned int sum_3d_array(unsigned int level);
void histoPyramidConstruction();
bool handleCudaError(const cudaError_t& status);
template<typename T>
T log2(T val);

bool runTests(unsigned char const * const voxels, const std::vector<std::pair<cudaExtent, cudaPitchedPtr> >& images_size_pointer, const int isolevel);

#endif // __gpu_mc_tests__
