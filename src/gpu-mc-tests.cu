/* see LICENSE file for copyright and license details */

#include "gpu-mc-tests.h"
#include <iostream>

// code to test classify cubes
// calculate a 1d index from a 3D position on a cube
unsigned int get_index(unsigned int x, unsigned int y, unsigned int z, size_t size) {
    return x + y*size + z*size*size;
}

// given a cubeid, calculate the position it is on the 3D cube
void get_voxel_from_cube_id(unsigned int cube_id, unsigned int *x, unsigned int *y, unsigned *z, size_t size) {
    // return lower left position of cube, other points can be obtained with +0,1
    *z = cube_id / (size-1) / (size-1);
    unsigned int cube_id_plane = cube_id % ((size-1) * (size-1));
    *y = cube_id_plane / (size-1);
    *x = cube_id_plane % (size-1);
}

// the cubeOffsets arent used linearly in the device code. this is the mapping
// to the used order
int bit2Offset[] = {0, 1, 3, 2, 4, 5, 7, 6};
uint4 lokalCubeOffsets[8] = {
    {0, 0, 0, 0},
    {1, 0, 0, 0},
    {0, 0, 1, 0},
    {1, 0, 1, 0},
    {0, 1, 0, 0},
    {1, 1, 0, 0},
    {0, 1, 1, 0},
    {1, 1, 1, 0},
};

// the number of triangles for each case
unsigned char lokalNrOfTriangles[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5, 2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3, 4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0};

// tests if the kernel updateScalarField worked correctly
bool testUpdateScalarField(unsigned char const * const voxels, const std::vector<std::pair<cudaExtent, cudaPitchedPtr> >& images_size_pointer, unsigned int& sum_of_triangles, int isolevel) {
    updateScalarField();
    // get level0 data from gpu
    uchar2 const * const lvl0_data = get_data_from_pitched_ptr<uchar2>(0);

    sum_of_triangles = 0;

    // calc for each voxel index and number of triangles using a different implementation
    const size_t size = images_size_pointer.at(0).first.depth;
    for (unsigned int i = 0; i < (size-1)*(size-1)*(size-1); i++) {
        // get base voxel of the cube
        unsigned int x, y, z;
        get_voxel_from_cube_id(i, &x, &y, &z, size);
        // look which vertices are below or above our threshold
        int lookuptable_index = 0;

        for (unsigned int id = 0; id < 8; id++) {
            uint4 offset = lokalCubeOffsets[bit2Offset[id]];
            unsigned char voxel = voxels[get_index(x + offset.x, y + offset.y, z + offset.z, size)];
            bool greater = voxel > isolevel;
            lookuptable_index |= greater << id;
        }
        unsigned int num_triangles = lokalNrOfTriangles[lookuptable_index];
        sum_of_triangles += num_triangles;

        // compare with results from gpu
        if (lookuptable_index != lvl0_data[get_index(x, y, z, size)].y) {
            std::cout << "No match at position: (" << x << ", " << y << ", " << z << ")" << std::endl;
            std::cout << "cube index calculated in software: " << lookuptable_index << "\ncube index calculated in hardware: " << static_cast<int>((lvl0_data[get_index(x, y, z, size)].y)) << std::endl;
            return false;
        }

        if (num_triangles != lvl0_data[get_index(x, y, z, size)].x) {
            std::cout << "No match in number of triangles at position: (" << x << ", " << y << ", " << z << ")" << std::endl;
            std::cout << "number triangles calculated in software: " << num_triangles << "\nnumber of triangles calculated in hardware: " << static_cast<int>((lvl0_data[get_index(x, y, z, size)].x)) << std::endl;
            return false;
        }
    }

    std::cout << "you will get " << sum_of_triangles << " triangles" << std::endl;

    delete [] lvl0_data;
    return true;
}
// end of code to test classifycubes

template<typename T>
bool templatedTestHistoPyramidConstruction(unsigned int level, const unsigned int& sum_of_triangles) {
    unsigned int sum = sum_3d_array<T>(level);
    if (sum != sum_of_triangles) {
        std::cout << "at level " << level << std::endl;
        std::cout << "number of triangles calculated in software and hardware mismatches!" << std::endl;
        std::cout << "software: " << sum_of_triangles << ", hardware: " << sum << std::endl;
    }
    return sum == sum_of_triangles;
}

// checks if number of triangles is equal to the sum calculated with
// testUpdateScalarField
bool testHistoPyramidConstruction(const std::vector<std::pair<cudaExtent, cudaPitchedPtr> >& images_size_pointer, const unsigned int& sum_of_triangles) {
    histoPyramidConstruction();
    bool success = true;
    for (unsigned int i = 0; i < log2(images_size_pointer.at(0).first.depth); i++) {
        if (i == 0)
            success &= templatedTestHistoPyramidConstruction<uchar2>(i, sum_of_triangles);
        else if (i == 1)
            success &= templatedTestHistoPyramidConstruction<uchar1>(i, sum_of_triangles);
        else if (i > 1 && i < 5)
            success &= templatedTestHistoPyramidConstruction<ushort1>(i, sum_of_triangles);
        else
            success &= templatedTestHistoPyramidConstruction<uint1>(i, sum_of_triangles);
    }
    return success;
}

// because of warnings during compilation, test if the cudaPitchedPtr, which are
// copied onto the gpu are valid and stay the same as if they are passed via
// parameters
bool testHistoPyramidTraversal(const std::vector<std::pair<cudaExtent, cudaPitchedPtr> >& images_size_pointer) {
    dim3 grid(1,1,1);
    dim3 block(1,1,1);
    bool h_success = true;
    bool * d_success;
    handleCudaError(cudaMalloc(&d_success, sizeof(bool)));
    bool success = true;
    unsigned int i = 0;
    for (std::vector<std::pair<cudaExtent, cudaPitchedPtr> >::const_iterator iter = images_size_pointer.begin(); iter != images_size_pointer.end(); iter++, i++) {
        std::pair<cudaExtent, cudaPitchedPtr> pair = *iter;
        cmp_pitched_ptr<<<grid, block>>>(i, pair.second, d_success);
        handleCudaError(cudaGetLastError());
        cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
        if (!h_success) {
            std::cout << "something is wrong with the cudaPitchedPtr copied to the GPU via cudaMemcpyToSymbol at level " << i << std::endl;
        }
        success &= h_success;
    }
    handleCudaError(cudaFree(d_success));
    return success;
}

bool runTests(unsigned char const * const voxels, const std::vector<std::pair<cudaExtent, cudaPitchedPtr> >& images_size_pointer, const int isolevel) {
  unsigned int sum_of_triangles;
  bool success = testUpdateScalarField(voxels, images_size_pointer, sum_of_triangles, isolevel);
  success &= testHistoPyramidConstruction(images_size_pointer, sum_of_triangles);
  success &= testHistoPyramidTraversal(images_size_pointer);

  return success;
}
