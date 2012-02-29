#include "gpu-mc.h"
#include "gpu-mc-kernel.h"
#include <utility>
#include <vector>
#include <iostream>
#include <cassert>
#include <cuda_gl_interop.h>

const unsigned int CUBESIZE = 8;
const unsigned int LOG2CUBESIZE = 3;
const unsigned int CUBESIZEHP = 2;
const unsigned int LOG2CUBESIZEHP = 1;
unsigned int SIZE;
unsigned int rawMemSize;
unsigned char * rawDataPtr;
struct cudaGraphicsResource * vbo_cuda = NULL;
GLuint vbo_gl;
size_t vbo_size = 0;

unsigned int sum_of_triangles = 0;

// TODO How to use the VBO:
//      1. calc number of triangles
//      2. resize VBO to the correct size (triangles and normals)
//      3. calc the triangles
//      4. render

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
        default: {
            error_msg = "unknown error";
            break;
        }
    }
    
    if (status != cudaSuccess)
        std::cout << error_msg << std::endl;

    return status != cudaSuccess;
}

void setupCuda(unsigned char * voxels, unsigned int size, GLuint vbo) {
    if (vbo == 0 && false) {
        std::cout << "Vertex Buffer Object to write into is invalid, I will exit" << std::endl;
        exit(1);
    }
    vbo_gl = vbo;
    handleCudaError(cudaGLSetGLDevice(0));

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

// classifies each voxel and calculates the number of triangles needed for this
// voxel
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
    }
}

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

void resizeVBO(size_t _vbo_size) {
    if (vbo_cuda != NULL) {
        handleCudaError(cudaGraphicsUnregisterResource(vbo_cuda));
        vbo_cuda = NULL;
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo_gl);
    glBufferData(GL_ARRAY_BUFFER, _vbo_size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    handleCudaError(cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo_gl, cudaGraphicsMapFlagsWriteDiscard));
    
    vbo_size = _vbo_size;
}

// creates the VBO
int histoPyramidTraversal() {
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

    for (unsigned int i = 0; i < num_of_levels; i++) {
        handleCudaError(cudaMemcpyToSymbol("levels", &(images_size_pointer.at(i).second), sizeof(cudaPitchedPtr), i*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice));
    }
    // TODO to get this working I need to setup OpenGL with VBO
    //      since OpenGL over SSH is hard, maybe I can just write into an array
    // resize buffer
    // normals, triangles, three coordinates, three points in float
    size_t buffer_size = sum*2*3*3*sizeof(float);
    // just increasing the buffer would be enough as well, but atm this is easier
    if (buffer_size > vbo_size)
        resizeVBO(buffer_size);

    float3 * triangle_data = NULL;
    cudaGraphicsMapResources(1, &vbo_cuda, 0);
    size_t num_bytes = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&triangle_data, &num_bytes, vbo_cuda);
    assert(num_bytes >= buffer_size);

    dim3 block(CUBESIZE, CUBESIZE, CUBESIZE);
    dim3 grid((pair.first.depth / CUBESIZE) * (pair.first.depth / CUBESIZE), pair.first.depth / CUBESIZE, 1);
    int log2GridSize = log2(pair.first.depth / CUBESIZE);
    
    traverseHP<<<grid, block>>>(
        triangle_data,
        isolevel,
        sum,
        log2GridSize, pair.first.depth/CUBESIZE-1, LOG2CUBESIZE, 
        pair.first.depth
        );
    
    cudaGraphicsUnmapResources(1, &vbo_cuda, 0);
    return sum;
}

#ifdef DEBUG
bool operator==(const cudaPitchedPtr& cpp1, const cudaPitchedPtr& cpp2) {
    return cpp1.pitch == cpp2.pitch && cpp1.ptr == cpp2.ptr && cpp1.xsize == cpp2.xsize && cpp1.ysize == cpp2.ysize;
}

bool testCudaPitchedPtrOnDevice() {
    std::pair<cudaExtent, cudaPitchedPtr> pair = images_size_pointer.back();
    dim3 grid(1,1,1);
    dim3 block(1,1,1);
    unsigned int h_count = 0;
    unsigned int * d_count;
    cudaMalloc(&d_count, sizeof(unsigned int));
    sum_values_of_pitched_ptr<<<grid, block>>>(d_count, pair.first.depth, log2(pair.first.depth));
    cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    bool success = sum_of_triangles == h_count;
    if (!success)
        std::cout << "something is wrong with the cudaPitchedPtr copied to the GPU via cudaMemcpyToSymbol" << std::endl;
    return success;
}

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
            std::cout << "cudaPitchedPtr into arguments of GPU does not match at level " << i << std::endl;
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
  return success;
}
#endif // DEBUG

int marching_cube(int _isolevel) {
    if (isolevel != _isolevel) {
        isolevel = _isolevel; 
        // first level
        updateScalarField();
        // all other levels
        histoPyramidConstruction();
        histoPyramidTraversal();
    }
    return sum_of_triangles;
}
