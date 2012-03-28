/* see LICENSE file for copyright and license details */

#include "gpu-mc.h"
#ifdef DEBUG
// gpu-mc.h must be before gpu-mc-tests.cu because gpu-mc.h defines the DEBUG
// makro. including a .cu is ugly, but I can't get it compiled and liked otherwise
#include "gpu-mc-tests.cu"
#endif
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
    // of vbo is zero, this is likely started via ssh
    vbo_gl = vbo;
    if (vbo != 0)
        handleCudaError(cudaGLSetGLDevice(0));

    // Create images for the HistogramPyramid
    cudaExtent bufferSize;
    cudaPitchedPtr tmpDataPtr;
    // Make the two first buffers use INT8
    // first buffer
    bufferSize.width = size * sizeof(uchar2);
    bufferSize.height = size;
    bufferSize.depth = size;
    handleCudaError(cudaMalloc3D(&tmpDataPtr, bufferSize));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));

    // second buffer
    bufferSize.width = bufferSize.depth/2 * sizeof(uchar1);
    bufferSize.height = bufferSize.depth/2;
    bufferSize.depth = bufferSize.depth/2;
    handleCudaError(cudaMalloc3D(&tmpDataPtr, bufferSize));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));

    // And the third, fourth and fifth INT16
    for (unsigned int i = 0; i < 3; i++) {
        bufferSize.width = bufferSize.depth/2 * sizeof(ushort1);
        bufferSize.height = bufferSize.depth/2;
        bufferSize.depth = bufferSize.depth/2;
        handleCudaError(cudaMalloc3D(&tmpDataPtr, bufferSize));
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    }

    // The rest will use INT32
    for(unsigned int i = 5; i < (log2(size)); i++) {
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
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    }

    // copy cudapitchedPtr to device
    // this will be used by the kernel histoPyramidTraversal(), because there is
    // an argument limit of 256bytes and 10 levels of cudaPitchedPtr would
    // exceed this limit
    if (images_size_pointer.size() > 10) {
        std::cout << "such large volumes aren't considered yet. but the changes are minimal to support them. just increase the array size of pitchedptr in the file gpu-mc-kernel.h and recompile. do not to forget to alter this test." << std::cout;
        exit(1);
    }
    for (unsigned int i = 0; i < images_size_pointer.size(); i++) {
        handleCudaError(cudaMemcpyToSymbol("levels", &(images_size_pointer.at(i).second), sizeof(cudaPitchedPtr), i*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice));
    }

    // Transfer dataset to device
    // size of the voxelvolume in bytes
    unsigned int rawMemSize = size*size*size*sizeof(unsigned char);
    handleCudaError(cudaMalloc((void **) &rawDataPtr, rawMemSize));
    handleCudaError(cudaMemcpy(rawDataPtr, voxels, rawMemSize, cudaMemcpyHostToDevice));

    #ifdef DEBUG
    if (!runTests(voxels, images_size_pointer, isolevel))
        std::cout << "something with the tests went wrong" << std::endl;
    std::cout << std::endl << std::endl << std::endl << std::endl;
    #endif
}

// classifies each voxel and calculates the number of triangles needed for this
// voxel
void updateScalarField() {
    cudaExtent _size = images_size_pointer.at(0).first;
    dim3 block(CUBESIZE, CUBESIZE, CUBESIZE);
    dim3 grid((_size.depth / CUBESIZE) * (_size.depth / CUBESIZE), _size.depth / CUBESIZE, 1);
    int log2GridSize = log2(_size.depth / CUBESIZE);
    kernelClassifyCubes<<<grid , block>>>(images_size_pointer.at(0).second, rawDataPtr, isolevel, log2GridSize, _size.depth/CUBESIZE-1, log2(CUBESIZE), _size.depth);
    handleCudaError(cudaGetLastError());
    handleCudaError(cudaThreadSynchronize());
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

// calculates the total number of triangles needed
void histoPyramidConstruction() {
    // when the histopyramid is created 8 cubes are added and the result will be
    // saved in a new cube on a smaller volume
    const unsigned int CUBESIZEHP = 2;
    dim3 block(CUBESIZEHP, CUBESIZEHP, CUBESIZEHP);
    
    // i=    0       1        2        3        4      5
    // uchar2, uchar1, ushort1, ushort1, ushort1, uint1, ...
    for (unsigned int i = 0; i < log2(images_size_pointer.at(0).first.depth)-1; i++) {
        cudaExtent _size = images_size_pointer.at(i+1).first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        if (i == 0)
            // second level
            // uchar2 -> uchar1
            kernelConstructHPLevel<uchar2, uchar1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, log2(CUBESIZEHP));
        else if (i == 1)
            // third level
            // uchar1 -> ushort1
            kernelConstructHPLevel<uchar1, ushort1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, log2(CUBESIZEHP)); 
        else if (i == 2 || i == 3)
            // fourth, fifth level
            // ushort1 -> ushort1
            // ushort1 -> ushort1
            kernelConstructHPLevel<ushort1, ushort1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, log2(CUBESIZEHP)); 
        else if (i == 4)
            // sixth level
            // ushort1 -> uint1
            kernelConstructHPLevel<ushort1, uint1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, log2(CUBESIZEHP)); 
        else
            // all other levels
            // uint1 -> uint1
            kernelConstructHPLevel<uint1, uint1><<<grid, block>>>(images_size_pointer.at(i).second , images_size_pointer.at(i+1).second, log2GridSize, _size.depth/CUBESIZEHP-1, log2(CUBESIZEHP));
        handleCudaError(cudaGetLastError());
        handleCudaError(cudaThreadSynchronize());
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
unsigned int getNumberOfTriangles() {
    unsigned int sum = 0;
    size_t num_of_levels = images_size_pointer.size();
    std::pair<cudaExtent, cudaPitchedPtr> pair =  images_size_pointer.back();
    if (num_of_levels == 1)
        sum = sum_3d_array<uchar2>(pair);
    else if (num_of_levels == 2)
        sum = sum_3d_array<uchar1>(pair);
    else if (num_of_levels == 3 || num_of_levels == 4 || num_of_levels == 5)
        sum = sum_3d_array<ushort1>(pair);
    else
        sum = sum_3d_array<uint1>(pair);
    sum_of_triangles = sum;
    std::cout << "you will get " << sum << " triangles" << std::endl;

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
    int grid_dim_x = ceil(sqrt(number_of_blocks));
    int grid_dim_y = ceil(number_of_blocks/grid_dim_x);
    dim3 grid(grid_dim_x, grid_dim_y, 1);
    
    size_t size = images_size_pointer.at(0).first.depth;
    traverseHP<<<grid, block>>>(
        rawDataPtr,
        triangle_data,
        isolevel,
        sum_of_triangles,
        log2(size),
        size,
        log2(tmp_cube_size)
        );
    handleCudaError(cudaGetLastError());
    handleCudaError(cudaThreadSynchronize());
    
    freeResources(triangle_data);
    return sum_of_triangles;
}

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
