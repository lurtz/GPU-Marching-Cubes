#include "gpu-mc.h"
#include "gpu-mc-kernel.h"
#include <utility>
#include <vector>
#include <iostream>

const unsigned int CUBESIZE = 8;
const unsigned int LOG2CUBESIZE = 3;
const unsigned int CUBESIZEHP = 2;
const unsigned int LOG2CUBESIZEHP = 1;
unsigned int SIZE;
unsigned int rawMemSize;
unsigned char * rawDataPtr;

#ifdef DEBUG
unsigned int sum_of_triangles = 0;
#endif

// first level has char4 as datatype, which contains: (number of triangles, cube index, value of first cube element, 0)
// first to second level contain volumes with unsigned char as elements
// third to fifth (including) level contain unsigned short as elements
// sixth level and more uses int
std::vector<std::pair<cudaExtent, cudaPitchedPtr> > images_size_pointer;

int isolevel = 50;

template<typename T>
T log2(T val) {
    T log2Val = 0;
    while (val > 1) {
      val /= 2; log2Val++;
    }
    return log2Val;
}

bool handleCudaError(const cudaError_t& status) {
    if (status == cudaErrorInvalidValue) {
        std::cout << "cudaErrorInvalidValue" << std::endl;
    }
    if (status == cudaErrorInvalidDevicePointer) {
        std::cout << "cudaErrorInvalidDevidePointer" << std::endl;
    }
    return status != cudaSuccess;
}

void setupCuda(unsigned char * voxels, unsigned int size) {
    SIZE = size;

    // Create images for the HistogramPyramid
    cudaExtent bufferSize;
    cudaPitchedPtr tmpDataPtr;
    // Make the two first buffers use INT8
    bufferSize.width = SIZE * sizeof(uchar4);
    bufferSize.height = SIZE;
    bufferSize.depth = SIZE;
    cudaMalloc3D(&tmpDataPtr, bufferSize);
    handleCudaError(cudaMemset3D(tmpDataPtr, 0, bufferSize));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));

    bufferSize.width = bufferSize.depth/2 * sizeof(uchar1);
    bufferSize.height = bufferSize.depth/2;
    bufferSize.depth = bufferSize.depth/2;
    cudaMalloc3D(&tmpDataPtr, bufferSize);
    handleCudaError(cudaMemset3D(tmpDataPtr, 0, bufferSize));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));

    // And the third, fourth and fifth INT16
    for (unsigned int i = 0; i < 3; i++) {
        bufferSize.width = bufferSize.depth/2 * sizeof(ushort1);
        bufferSize.height = bufferSize.depth/2;
        bufferSize.depth = bufferSize.depth/2;
        cudaMalloc3D(&tmpDataPtr, bufferSize);
        handleCudaError(cudaMemset3D(tmpDataPtr, 0, bufferSize));
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    }

    // The rest will use INT32
    for(unsigned int i = 5; i < (log2(SIZE)); i++) {
        bufferSize.width = bufferSize.depth/2 * sizeof(uint1);
        bufferSize.height = bufferSize.depth/2;
        bufferSize.depth = bufferSize.depth/2;
        // Image cant be 1x1x1
        if (bufferSize.depth == 1) {
            bufferSize.width = 2 * sizeof(uint1);
            bufferSize.height = 2;
            bufferSize.depth = 2;
        }
        cudaMalloc3D(&tmpDataPtr, bufferSize);
        handleCudaError(cudaMemset3D(tmpDataPtr, 0, bufferSize));
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    }

    // Transfer dataset to device
    rawMemSize = SIZE*SIZE*SIZE*sizeof(unsigned char);
    cudaMalloc((void **) &rawDataPtr, rawMemSize);
    cudaMemcpy(rawDataPtr, voxels, rawMemSize, cudaMemcpyHostToDevice);
//    delete[] voxels;
}

void updateScalarField() {
    cudaExtent _size = images_size_pointer.at(0).first;
    dim3 block(CUBESIZE, CUBESIZE, CUBESIZE);
    dim3 grid((_size.depth / CUBESIZE) * (_size.depth / CUBESIZE), _size.depth / CUBESIZE, 1);
    int log2GridSize = log2(_size.depth / CUBESIZE);
    kernelClassifyCubes<<<grid , block>>>(images_size_pointer.at(0).second, rawDataPtr, isolevel, log2GridSize, _size.depth/CUBESIZE-1, LOG2CUBESIZE, _size.depth);
}

#ifdef DEBUG
// code to test classify cubes
unsigned int get_index(unsigned int x, unsigned int y, unsigned int z) {
  return x + y*SIZE + z*SIZE*SIZE;
}

void get_voxel_from_cube_id(unsigned int cube_id, unsigned int *x, unsigned int *y, unsigned *z) {
  // return lower left position of cube, other points can be obtained with +0,1
  *z = cube_id / (SIZE-1) / (SIZE-1);
  unsigned int cube_id_plane = cube_id % ((SIZE-1) * (SIZE-1));
  *y = cube_id_plane / (SIZE-1);
  *x = cube_id_plane % (SIZE-1);
}

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

unsigned char lokalNrOfTriangles[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5, 2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3, 4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0};

template<typename T>
T* get_data_from_pitched_ptr(cudaExtent size, cudaPitchedPtr source) {
    T * lvl0_data = new T[size.depth*size.depth*size.depth];
    cudaPitchedPtr h_pitched_ptr = make_cudaPitchedPtr(lvl0_data, size.depth*sizeof(T), size.depth, size.depth);
    struct cudaMemcpy3DParms parms = {0};
    parms.srcPtr = source;
    parms.dstPtr = h_pitched_ptr;
    parms.extent = size;
    parms.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&parms);
    return lvl0_data;
}

template<typename T>
T* get_data_from_pitched_ptr(std::pair<cudaExtent, cudaPitchedPtr> source) {
    return get_data_from_pitched_ptr<T>(source.first, source.second);
}

template<typename T>
T* get_data_from_pitched_ptr(unsigned int level) {
    return get_data_from_pitched_ptr<T>(images_size_pointer.at(level));
}

bool testUpdateScalarField(unsigned char * voxels) {
    // get level0 data from gpu
    uchar4 * lvl0_data = get_data_from_pitched_ptr<uchar4>(0);

    sum_of_triangles = 0;

    // calc for each voxel index and number of triangles using a different implementation
    for (unsigned int i = 0; i < (SIZE-1)*(SIZE-1)*(SIZE-1); i++) {
        // get base voxel of the cube
        unsigned int x, y, z;
        get_voxel_from_cube_id(i, &x, &y, &z);  
        // look which vertices are below or above our threshold
        int lookuptable_index = 0;

        for (unsigned int id = 0; id < 8; id++) {
            uint4 offset = lokalCubeOffsets[bit2Offset[id]];
            unsigned char voxel = voxels[get_index(x + offset.x, y + offset.y, z + offset.z)];
            bool greater = voxel > isolevel;
            lookuptable_index |= greater << id;
        }
        unsigned int num_triangles = lokalNrOfTriangles[lookuptable_index];
        sum_of_triangles += num_triangles;

        // compare with results from gpu
        if (voxels[get_index(x, y, z)] != lvl0_data[get_index(x, y, z)].z) {
            std::cout << "No match at position: (" << x << ", " << y << ", " << z << ")" << std::endl;
            std::cout << "vertex value got from raw data is: " << static_cast<int>(voxels[get_index(x, y, z)]) << "\n value saved on gpu is: " << static_cast<int>(lvl0_data[get_index(x, y, z)].z) << std::endl;
            return false;
        }

        if (lookuptable_index != lvl0_data[get_index(x, y, z)].y) {
            std::cout << "No match at position: (" << x << ", " << y << ", " << z << ")" << std::endl;
            std::cout << "cube index calculated in software: " << lookuptable_index << "\ncube index calculated in hardware: " << static_cast<int>((lvl0_data[get_index(x, y, z)].y)) << std::endl;
            return false;
        }

        if (num_triangles != lvl0_data[get_index(x, y, z)].x) {
            std::cout << "No match in number of triangles at position: (" << x << ", " << y << ", " << z << ")" << std::endl;
            std::cout << "number triangles calculated in software: " << num_triangles << "\nnumber of triangles calculated in hardware: " << static_cast<int>((lvl0_data[get_index(x, y, z)].x)) << std::endl;
            return false;
        }
    }

    std::cout << "you will get " << sum_of_triangles << " triangles" << std::endl;

    delete [] lvl0_data;
    return true;
}
// end of code to test classifycubes
#endif // DEBUG

void histoPyramidConstruction() {
    // first level
    updateScalarField();

    dim3 block(CUBESIZEHP, CUBESIZEHP, CUBESIZEHP);
    
    for (unsigned int i = 0; i < log2(SIZE)-1; i++) {
        cudaExtent _size = images_size_pointer.at(i+1).first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        if (i == 0)
            // second level
            kernelConstructHPLevel<uchar4, uchar1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP);
        else if (i == 1)
            // third level
            kernelConstructHPLevel<uchar1, ushort1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
        else if (i > 1 && i < 4)
            // fourth, fifth level
            kernelConstructHPLevel<ushort1, ushort1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
        else if (i == 5)
            // sixth level
            kernelConstructHPLevel<ushort1, uint1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
        else
            // all other levels
            kernelConstructHPLevel<uint1, uint1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
    }
}

#ifdef DEBUG
template<typename T>
bool templatedTestHistoPyramidConstruction(unsigned int level) {
    std::pair<cudaExtent, cudaPitchedPtr> pair = images_size_pointer.at(level);
    T* sum_of_triangles_from_gpu = get_data_from_pitched_ptr<T>(pair);
    unsigned int sum = 0;
    for (unsigned int id = 0; id < pair.first.depth*pair.first.depth*pair.first.depth; id++) {
        sum+= sum_of_triangles_from_gpu[id].x;
    }
    delete [] sum_of_triangles_from_gpu;
    sum_of_triangles_from_gpu = 0;
    if (sum != sum_of_triangles) {
        std::cout << "at level " << level << std::endl;
        std::cout << "number of triangles calculated in software and hardware mismatches!" << std::endl;
        std::cout << "software: " << sum_of_triangles << ", hardware: " << sum << std::endl;
    }
    return sum == sum_of_triangles;
}

bool testHistoPyramidConstruction() {
    bool success = true;
    for (unsigned int i = 0; i < log2(SIZE); i++) {
        if (i == 0)
            success &= templatedTestHistoPyramidConstruction<uchar4>(i);
        else if (i == 1)
            success &= templatedTestHistoPyramidConstruction<uchar1>(i);
        else if (i > 1 && i < 5)
            success &= templatedTestHistoPyramidConstruction<ushort1>(i);
        else
            success &= templatedTestHistoPyramidConstruction<uint1>(i);
    }
    return success;
}
#endif // DEBUG
