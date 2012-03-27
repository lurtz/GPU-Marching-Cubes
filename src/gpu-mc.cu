/* see LICENSE file for copyright and license details */

#include "gpu-mc.h"
#include "gpu-mc-kernel.h"
#include <utility>
#include <vector>
#include <iostream>
#include <cassert>
#include <cuda_gl_interop.h>
#include <cmath>
#include <ctime>

// the sidelength of a block on the grid, it is always cubic, which results to
// 8*8*8 = 512 threads per block
const unsigned int CUBESIZE = 8;
const unsigned int LOG2CUBESIZE = 3;
// when the histopyramid is created 8 cubes are added and the result will be
// saved in a new cube on a smaller volume
const unsigned int CUBESIZEHP = 2;
const unsigned int LOG2CUBESIZEHP = 1;
// the size of the voxelvolume TODO not needed
unsigned int SIZE;
// size of the voxelvolume in bytes
unsigned int rawMemSize;
// pointer of voxelvolume on device memory
unsigned char * rawDataPtr;
// these will be used with opengl interop
// the triangulated mesh will be written into a VBO
struct cudaGraphicsResource * vbo_cuda = NULL;
GLuint vbo_gl = 0;
// used to check if the VBO is big enough for the mesh
size_t vbo_size = 0;
// how much triangles will be constructed
unsigned int sum_of_triangles = 0;

// How to use the VBO:
//      1. calc number of triangles
//      2. resize VBO to the correct size (triangles and normals)
//      3. calc the triangles
//      4. render

// first level has char4 as datatype, which contains: (number of triangles, cube index, value of first cube element, 0)
// first to second level contain volumes with unsigned char as elements
// third to fifth (including) level contain unsigned short as elements
// sixth level and more uses int
// this vector saves for each level of the histopyramid its size and the
// pointer to device memory. level 0 is the largest with the most voxels
std::vector<std::pair<cudaExtent, cudaPitchedPtr> > images_size_pointer;

// initial isolevel
int isolevel = 49;

// slow variant of getting the binary logarithm
template<typename T>
T log2(T val) {
    T log2Val = 0;
    while (val > 1) {
      val /= 2; log2Val++;
    }
    return log2Val;
}

// check if something went wrong
bool handleCudaError(const cudaError_t& status) {
    std::string error_msg;
    switch (status) {
        case cudaErrorInvalidValue: {
            error_msg = "cudaErrorInvalidValue";
            break;
        }
        case cudaErrorInvalidDevicePointer: {
            error_msg = "cudaErrorInvalidDevidePointer";
            break;
        }
        case cudaErrorInvalidSymbol: {
            error_msg = "cudaErrorInvalidSymbol";
            break;
        }
        case cudaErrorInvalidMemcpyDirection: {
            error_msg = "cudaErrorInvalidMemcpyDirection";
            break;
        }
        case cudaErrorInvalidDevice: {
            error_msg = "cudaErrorInvalidDevice";
            break;
        }
        case cudaErrorSetOnActiveProcess: {
            error_msg = "cudaErrorSetOnActiveProcess";
            break;
        }
        case cudaErrorInvalidResourceHandle: {
            error_msg = "cudaErrorInvalidResourceHandle";
            break;
        }
        case cudaErrorMemoryAllocation: {
            error_msg = cudaErrorMemoryAllocation;
            break;
        }
        case cudaErrorUnknown: {
            error_msg = "cudaErrorUnknown";
            break;
        }
        case cudaErrorLaunchTimeout: {
            error_msg = "cudaErrorLaunchTimeout";
            break;
        }
        case cudaErrorNoDevice: {
            error_msg = "cudaErrorNoDevice";
            break;
        }
        case cudaSuccess: {
            break;
        }
        case cudaErrorLaunchFailure: {
            error_msg = "cudaErrorLaunchFailure";
            break;
        }
        case cudaErrorInvalidDeviceFunction: {
            error_msg = "cudaErrorInvalidDeviceFunction";
            break;
        }
        case cudaErrorLaunchOutOfResources: {
            error_msg= "cudaErrorLaunchOutOfResources";
            break;
        }
        default: {
            error_msg = "unknown error";
            break;
        }
    }
    
    if (status != cudaSuccess) {
        std::cout << "!!!!!!!!! " << error_msg << ", " << cudaGetErrorString(status) << " !!!!!!!!!" << std::endl;
        exit(1);
    }

    return status != cudaSuccess;
}

void setupCuda(unsigned char * voxels, unsigned int size, GLuint vbo) {
    // TODO get rid of cudaMemset3D()
    // TODO keep only one copy of the voxels in device memory
    // of vbo is zero, this is likely started via ssh
    vbo_gl = vbo;
    if (vbo != 0)
        handleCudaError(cudaGLSetGLDevice(0));

    SIZE = size;

    // Create images for the HistogramPyramid
    cudaExtent bufferSize;
    cudaPitchedPtr tmpDataPtr;
    // Make the two first buffers use INT8
    // first buffer
    bufferSize.width = SIZE * sizeof(uchar4);
    bufferSize.height = SIZE;
    bufferSize.depth = SIZE;
    handleCudaError(cudaMalloc3D(&tmpDataPtr, bufferSize));
    handleCudaError(cudaMemset3D(tmpDataPtr, 0, bufferSize));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));

    // second buffer
    bufferSize.width = bufferSize.depth/2 * sizeof(uchar1);
    bufferSize.height = bufferSize.depth/2;
    bufferSize.depth = bufferSize.depth/2;
    handleCudaError(cudaMalloc3D(&tmpDataPtr, bufferSize));
    handleCudaError(cudaMemset3D(tmpDataPtr, 0, bufferSize));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));

    // And the third, fourth and fifth INT16
    for (unsigned int i = 0; i < 3; i++) {
        bufferSize.width = bufferSize.depth/2 * sizeof(ushort1);
        bufferSize.height = bufferSize.depth/2;
        bufferSize.depth = bufferSize.depth/2;
        handleCudaError(cudaMalloc3D(&tmpDataPtr, bufferSize));
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
        handleCudaError(cudaMalloc3D(&tmpDataPtr, bufferSize));
        handleCudaError(cudaMemset3D(tmpDataPtr, 0, bufferSize));
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    }

    // copy cudapitchedPtr to device
    // this will be used by the kernel histoPyramidTraversal(), because there is
    // an argument limit of 256bytes and 10 levels of cudaPitchedPtr would
    // exceed this limit
    for (unsigned int i = 0; i < images_size_pointer.size(); i++) {
        handleCudaError(cudaMemcpyToSymbol("levels", &(images_size_pointer.at(i).second), sizeof(cudaPitchedPtr), i*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice));
    }

    // Transfer dataset to device
    rawMemSize = SIZE*SIZE*SIZE*sizeof(unsigned char);
    handleCudaError(cudaMalloc((void **) &rawDataPtr, rawMemSize));
    handleCudaError(cudaMemcpy(rawDataPtr, voxels, rawMemSize, cudaMemcpyHostToDevice));
}

// classifies each voxel and calculates the number of triangles needed for this
// voxel
void updateScalarField() {
    cudaExtent _size = images_size_pointer.at(0).first;
    dim3 block(CUBESIZE, CUBESIZE, CUBESIZE);
    dim3 grid((_size.depth / CUBESIZE) * (_size.depth / CUBESIZE), _size.depth / CUBESIZE, 1);
    int log2GridSize = log2(_size.depth / CUBESIZE);
    kernelClassifyCubes<<<grid , block>>>(images_size_pointer.at(0).second, rawDataPtr, isolevel, log2GridSize, _size.depth/CUBESIZE-1, LOG2CUBESIZE, _size.depth);
    handleCudaError(cudaGetLastError());
    cudaThreadSynchronize();
}

// copies data from device memory into an array on host memory and returns a 
// pointer to the array on host memory. don't forget to delete the array
template<typename T>
T* get_data_from_pitched_ptr(cudaExtent size, cudaPitchedPtr source) {
    T * lvl0_data = new T[size.depth*size.depth*size.depth];
    cudaPitchedPtr h_pitched_ptr = make_cudaPitchedPtr(lvl0_data, size.depth*sizeof(T), size.depth, size.depth);
    struct cudaMemcpy3DParms parms = {0};
    parms.srcPtr = source;
    parms.dstPtr = h_pitched_ptr;
    parms.extent = size;
    parms.kind = cudaMemcpyDeviceToHost;
    handleCudaError(cudaMemcpy3D(&parms));
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

#ifdef DEBUG
// code to test classify cubes
// calculate a 1d index from a 3D position on a cube
unsigned int get_index(unsigned int x, unsigned int y, unsigned int z) {
  return x + y*SIZE + z*SIZE*SIZE;
}

// given a cubeid, calculate the position it is on the 3D cube
void get_voxel_from_cube_id(unsigned int cube_id, unsigned int *x, unsigned int *y, unsigned *z) {
  // return lower left position of cube, other points can be obtained with +0,1
  *z = cube_id / (SIZE-1) / (SIZE-1);
  unsigned int cube_id_plane = cube_id % ((SIZE-1) * (SIZE-1));
  *y = cube_id_plane / (SIZE-1);
  *x = cube_id_plane % (SIZE-1);
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
bool testUpdateScalarField(unsigned char * voxels) {
    updateScalarField();
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

// calculates the total number of triangles needed
void histoPyramidConstruction() {
    dim3 block(CUBESIZEHP, CUBESIZEHP, CUBESIZEHP);
    
    // i=    0       1        2        3        4      5
    // uchar4, uchar1, ushort1, ushort1, ushort1, uint1, ...
    for (unsigned int i = 0; i < log2(SIZE)-1; i++) {
        cudaExtent _size = images_size_pointer.at(i+1).first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        if (i == 0)
            // second level
            // uchar4 -> uchar1
            kernelConstructHPLevel<uchar4, uchar1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP);
        else if (i == 1)
            // third level
            // uchar1 -> ushort1
            kernelConstructHPLevel<uchar1, ushort1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
        else if (i == 2 || i == 3)
            // fourth, fifth level
            // ushort1 -> ushort1
            // ushort1 -> ushort1
            kernelConstructHPLevel<ushort1, ushort1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
        else if (i == 4)
            // sixth level
            // ushort1 -> uint1
            kernelConstructHPLevel<ushort1, uint1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
        else
            // all other levels
            // uint1 -> uint1
            kernelConstructHPLevel<uint1, uint1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
        handleCudaError(cudaGetLastError());
        cudaThreadSynchronize();
    }
}

// sums the data of a 3D array. this will mainly be used to get the number of
// triangles from the top level of the histoPyramid
template<typename T>
unsigned int sum_3d_array(T const * const _3darray, const cudaExtent& dim) {
    unsigned int sum = 0;
    for (unsigned int id = 0; id < dim.depth*dim.depth*dim.depth; id++) {
        sum+= _3darray[id].x;
    }
    return sum;
}

template<typename T>
unsigned int sum_3d_array(const std::pair<cudaExtent, cudaPitchedPtr>& pair) {
    T* sum_of_triangles_from_gpu = get_data_from_pitched_ptr<T>(pair);
    unsigned int sum = sum_3d_array(sum_of_triangles_from_gpu, pair.first);
    delete [] sum_of_triangles_from_gpu;
    return sum;
}

template<typename T>
unsigned int sum_3d_array(unsigned int level) {
    std::pair<cudaExtent, cudaPitchedPtr> pair = images_size_pointer.at(level);
    return sum_3d_array<T>(pair);
}

#ifdef DEBUG
template<typename T>
bool templatedTestHistoPyramidConstruction(unsigned int level) {
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
bool testHistoPyramidConstruction() {
    histoPyramidConstruction();
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

// increases vbo size, if more triangles will be written
void resizeVBO(size_t _vbo_size, bool clear) {
    if (vbo_cuda != NULL) {
        handleCudaError(cudaGraphicsUnregisterResource(vbo_cuda));
        vbo_cuda = NULL;
    }

    float3 * data = NULL;
    if (clear) {
	// fill with zeros
        data = new float3[_vbo_size/sizeof(float3)];
        for (unsigned int i = 0; i < _vbo_size/sizeof(float3); i++) {
            float3 val = {0};
            data[i] = val;
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo_gl);
    glBufferData(GL_ARRAY_BUFFER, _vbo_size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // bind vbo with cuda
    handleCudaError(cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo_gl, cudaGraphicsMapFlagsWriteDiscard));

    delete [] data;
    
    vbo_size = _vbo_size;
}

// checks if vbo allocated by opengl is big enough to store all triangles
size_t resizeVBOIfNeeded(bool clear = false) {
    // resize buffer
    // normals, triangles, three coordinates, three points in float
    size_t buffer_size = sum_of_triangles*2*3*3*sizeof(float);
    if (buffer_size > vbo_size && vbo_gl != 0)
        resizeVBO(buffer_size, clear);
    return buffer_size;
}

// sums the top level of the histoPyramid and writes the number of levels used to the gpu
// TODO setting the number of levels if not needed
unsigned int getNumberOfTriangles() {
    unsigned int sum = 0;
    assert(log2(SIZE) == images_size_pointer.size());
    size_t num_of_levels = images_size_pointer.size();
    std::pair<cudaExtent, cudaPitchedPtr> pair =  images_size_pointer.back();
    if (num_of_levels == 1)
        sum = sum_3d_array<uchar4>(pair);
    else if (num_of_levels == 2)
        sum = sum_3d_array<uchar1>(pair);
    else if (num_of_levels == 3 || num_of_levels == 4 || num_of_levels == 5)
        sum = sum_3d_array<ushort1>(pair);
    else
        sum = sum_3d_array<uint1>(pair);
    sum_of_triangles = sum;
    std::cout << "you will get " << sum << " triangles" << std::endl;

    handleCudaError(cudaMemcpyToSymbol("num_of_levels", &num_of_levels, sizeof(size_t), 0, cudaMemcpyHostToDevice));
    return sum;
}

// returns a pointer to either the memory of the vbo or a cudaarray, if this is
// run without graphics output
float3 * getTriangleDataPointer() {
    size_t buffer_size = resizeVBOIfNeeded();

    float3 * triangle_data = NULL;
    if (vbo_gl != 0) {
        handleCudaError(cudaGraphicsMapResources(1, &vbo_cuda, 0));
        size_t num_bytes = 0;
        handleCudaError(cudaGraphicsResourceGetMappedPointer((void**)&triangle_data, &num_bytes, vbo_cuda));
        assert(num_bytes >= buffer_size);
    } else {
        handleCudaError(cudaMalloc(&triangle_data, buffer_size));
    }

    return triangle_data;
}

// makes the vbo available to opengl again or frees the cudaarray
void freeResources(float3 * triangle_data) {
    if (vbo_gl != 0)
        handleCudaError(cudaGraphicsUnmapResources(1, &vbo_cuda, 0));
    else
        handleCudaError(cudaFree(triangle_data));
}

// creates the VBO
// walk down the histopyramid to the base level. each thread gets a triangle, which he has to create and store in the vbo
int histoPyramidTraversal() {
    getNumberOfTriangles();

    float3 * triangle_data = getTriangleDataPointer();
    assert(triangle_data != NULL);

    // TODO ask device properties how much threads can be started
    //      there are cards, where not all 512 threads can be started, when
    //      some data is in memory
    unsigned int tmp_cube_size = CUBESIZE/2;
    dim3 block(tmp_cube_size, tmp_cube_size, tmp_cube_size);

    float number_of_blocks = static_cast<float>(sum_of_triangles)/tmp_cube_size/tmp_cube_size/tmp_cube_size;
    float sqrt_blocks = sqrt(number_of_blocks);
    // TODO there seems to be more blocks needed, but don't know what with the math is wrong
    //      when I add +1 to grid_dim_x it works with sphere
    int grid_dim_x = floor(sqrt_blocks);
    int grid_dim_y = ceil(sqrt_blocks);
    dim3 grid(grid_dim_x, grid_dim_y, 1);
    
    traverseHP<<<grid, block>>>(
        triangle_data,
        isolevel,
        sum_of_triangles,
        log2(SIZE),
        SIZE,
        log2(tmp_cube_size)
        );
    handleCudaError(cudaGetLastError());
    cudaThreadSynchronize();
    
    freeResources(triangle_data);
    return sum_of_triangles;
}

#ifdef DEBUG
// because of warnings during compilation, test if the cudaPitchedPtr, which are
// copied onto the gpu are valid and stay the same as if they are passed via
// parameters
bool testCudaPitchedPtrOnDevice() {
    dim3 grid(1,1,1);
    dim3 block(1,1,1);
    bool h_success = true;
    bool * d_success;
    handleCudaError(cudaMalloc(&d_success, sizeof(bool)));
    bool success = true;
    unsigned int i = 0;
    for (std::vector<std::pair<cudaExtent, cudaPitchedPtr> >::iterator iter = images_size_pointer.begin(); iter != images_size_pointer.end(); iter++, i++) {
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

// doesn't really test the kernel. just tests some things which are used by the
// kernel. the kernel was tested with opengl
bool testHistoPyramidTraversal() {
    histoPyramidTraversal();
    bool success = true;
    size_t num_of_levels_readback = 0;
    handleCudaError(cudaMemcpyFromSymbol(&num_of_levels_readback, "num_of_levels", sizeof(size_t), 0, cudaMemcpyDeviceToHost));

    success &= images_size_pointer.size() == num_of_levels_readback;
    if (!success) {
        std::cout << "number of levels on GPU are not set correctly: " << num_of_levels_readback << ", should be: " << images_size_pointer.size() << std::endl;
    }

    cudaPitchedPtr cpp = {0};
    for (unsigned int i = 0; i < num_of_levels_readback; i++) {
        handleCudaError(cudaMemcpyFromSymbol(&cpp, "levels", sizeof(cudaPitchedPtr), i*sizeof(cudaPitchedPtr), cudaMemcpyDeviceToHost));
        bool tmp_success = cpp == images_size_pointer.at(i).second;
        if (!tmp_success) {
            std::cout << "cudaPitchedPtr used as a argument for a kernel on the GPU does not match at level " << i << std::endl;
        }
        success &= tmp_success;
    }
    success &= testCudaPitchedPtrOnDevice();
    return success;
}


bool runTests(unsigned char * voxels) {
  bool success = testUpdateScalarField(voxels);
  success &= testHistoPyramidConstruction();
  success &= testHistoPyramidTraversal();
//  cudaDeviceReset();

  return success;
}
#endif // DEBUG

int marching_cube(int _isolevel) {
    if (isolevel != _isolevel) {
        isolevel = _isolevel; 
        clock_t start = clock();
        // first level
        updateScalarField();
        std::cout << "updateScalarField took " << static_cast<double>(clock()-start)/CLOCKS_PER_SEC << " seconds\n";

        // all other levels
        start = clock();
        histoPyramidConstruction();
        std::cout << "histoPyramidConstruction took " << static_cast<double>(clock()-start)/CLOCKS_PER_SEC << " seconds\n";

        start = clock();
        histoPyramidTraversal();
        std::cout << "histoPyramidTraversal took " << static_cast<double>(clock()-start)/CLOCKS_PER_SEC << " seconds\n";
    }
    return sum_of_triangles;
}
