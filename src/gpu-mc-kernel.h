/* see LICENSE file for copyright and license details */

#ifndef gpu_mc_kernel_h__
#define gpu_mc_kernel_h__

#include <device_functions.h>
#include "gpu-mc-lookuptables.h"

// tells histogramtraversal how much levels we have and thus how much parameters it has
__constant__ cudaPitchedPtr levels[10];

// converts the coordinates of the block on the grid and the position of the
// thread in the block into a position of the voxelvolume. each thread shall 
// get its unique position
// this only works with volumesizes and blocksizes as a power of 2
// log2* and mask are used to do multiplikation and computation of the remainder
__device__ uint4 getPosFromGrid(const uint3 blockIndex, const uint3 threadIndex, int log2BlockWidth, int mask, int log2CubeWidth) {
    int blkX = blockIndex.y;
    int blkY = blockIndex.x >> log2BlockWidth;
    int blkZ = blockIndex.x & mask;

    blkX = blkX << log2CubeWidth;
    blkY = blkY << log2CubeWidth;
    blkZ = blkZ << log2CubeWidth;

    int x = blkX + threadIndex.x;
    int y = blkY + threadIndex.y;
    int z = blkZ + threadIndex.z;

    return make_uint4(x, y, z, 0);
}

// calculates the array index of a 3D position in the voxeldatda
__device__ unsigned int getId(uint4 pos, int log2BlockWidth, int log2CubeWidth) {
    int log2Sum = log2BlockWidth + log2CubeWidth;
    return (((pos.z << log2Sum) + pos.y) << log2Sum) + pos.x;
}

// addition
inline __host__ __device__ uchar4 operator+(uchar4 a, uchar4 b) {
    return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// addition
inline __host__ __device__ uint4 operator+(uint4 a, uint4 b) {
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// multiplication
inline __host__ __device__ uint4 operator*(uint4 a, int b) {
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// subtraction
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ uint4 operator+=(uint4& a, const uint4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

// returns the pointer to a voxel inside a cuda 3D array
template<typename T>
inline __device__ T* get_voxel_address(cudaPitchedPtr cpptr, uint4 pos, int log2Size) {
    char* devPtr = (char*)cpptr.ptr;
    size_t pitch = cpptr.pitch;
    size_t slicePitch = pitch << log2Size;
    char* slice = devPtr + pos.z * slicePitch;
    T* row = (T*)(slice + pos.y * pitch);
    return &row[pos.x];
}

// returns the value of a voxel inside a cuda 3D array
template<typename T>
inline __device__ T get_voxel(cudaPitchedPtr cpptr, uint4 pos, int log2Size) {
    return *get_voxel_address<T>(cpptr, pos, log2Size);
}

// writes a value a voxel inside a cuda 3D array
template<typename T>
inline __device__ void write_voxel(cudaPitchedPtr cpptr, uint4 pos, int log2Size, T value) {
    *get_voxel_address<T>(cpptr, pos, log2Size) = value;
}

// interpolate between x and y if a ranges from 0 to 1
__device__ float3 mix(float3 x, float3 y, float a) {
  float3 diff = y-x;
  float3 scaled = diff*a;
  float3 ret_val = x + scaled;
  return ret_val;
}

// classifies each cube and computates a lookuptable index according to its 8 
// voxels and the isolevel. for each cube a thread is started.
// writes into a uchar4 volume with the following layout: nr of triangles for this cube, lookuptable index, value of voxel with id 0, nothing 
__global__ void kernelClassifyCubes(cudaPitchedPtr histoPyramid, unsigned char * rawData, int isolevel, int log2BlockWidth, int mask, int log2CubeWidth, unsigned int volumeSize) {
    // 2d grid
    // 3d kernel
    uint4 pos = getPosFromGrid(blockIdx, threadIdx, log2BlockWidth, mask, log2CubeWidth);
    
    unsigned int id = getId(pos, log2BlockWidth, log2CubeWidth);
    unsigned char first = rawData[id];

    // was not done in the opencl programm, maybe not needed
    // avoid looking over array boundaries
    uchar4 ret_val = make_uchar4(0, 0, first, 0);
    if (!(pos.x+1 >= volumeSize || pos.y+1 >= volumeSize || pos.z+1 >= volumeSize)) {
        // Find cube class nr
        const unsigned char cubeindex = 
            ((first > isolevel)) |
            ((rawData[getId(pos + cubeOffsets[1], log2BlockWidth, log2CubeWidth)] > isolevel) << 1) |
            ((rawData[getId(pos + cubeOffsets[3], log2BlockWidth, log2CubeWidth)] > isolevel) << 2) |
            ((rawData[getId(pos + cubeOffsets[2], log2BlockWidth, log2CubeWidth)] > isolevel) << 3) |
            ((rawData[getId(pos + cubeOffsets[4], log2BlockWidth, log2CubeWidth)] > isolevel) << 4) |
            ((rawData[getId(pos + cubeOffsets[5], log2BlockWidth, log2CubeWidth)] > isolevel) << 5) |
            ((rawData[getId(pos + cubeOffsets[7], log2BlockWidth, log2CubeWidth)] > isolevel) << 6) |
            ((rawData[getId(pos + cubeOffsets[6], log2BlockWidth, log2CubeWidth)] > isolevel) << 7);
        ret_val = make_uchar4(nrOfTriangles[cubeindex], cubeindex, first, 0);
    }

    // Store number of triangles
    write_voxel<uchar4>(histoPyramid, pos, log2BlockWidth+log2CubeWidth, ret_val);
}

// now tested and seems to work
// sums the contents of a 2x2x2 volume and stores the sum in a cell of a volume
// with half side lenghts.
template<typename T, typename Z>
__global__ void kernelConstructHPLevel(cudaPitchedPtr readHistoPyramid, cudaPitchedPtr writeHistoPyramid, int log2BlockWidth, int mask, int log2CubeWidth) {
    uint4 writePos = getPosFromGrid(blockIdx, threadIdx, log2BlockWidth, mask, log2CubeWidth);
    uint4 readPos = writePos*2;

    char* devPtr = (char*)readHistoPyramid.ptr;
    size_t pitch = readHistoPyramid.pitch;
    // source array has double side lengths
    size_t slicePitch = pitch << (log2BlockWidth + 1 + log2CubeWidth);
    Z writeValue = {0};
    for (unsigned int i = 0; i < 8; i++) {
        uint4 tmpPos = readPos+cubeOffsets[i];
        char* slice = devPtr + tmpPos.z * slicePitch;
        T* row = (T*)(slice + tmpPos.y * pitch);
        // first level has uchar4 as datatype
        writeValue.x += row[tmpPos.x].x;
    }

    write_voxel<Z>(writeHistoPyramid, writePos, log2BlockWidth + log2CubeWidth, writeValue);
}

// walks the histopyramid down one level. see the paper in the doc directory for
// how this is supposed to work.
template<typename T>
__device__ uint4 scanHPLevel(int target, __const__ cudaPitchedPtr hp, uint4 current, int log2Size) {
    int neighbors[8] = {
            get_voxel<T>(hp, current + cubeOffsets[0], log2Size).x,
            get_voxel<T>(hp, current + cubeOffsets[1], log2Size).x,
            get_voxel<T>(hp, current + cubeOffsets[2], log2Size).x,
            get_voxel<T>(hp, current + cubeOffsets[3], log2Size).x,
            get_voxel<T>(hp, current + cubeOffsets[4], log2Size).x,
            get_voxel<T>(hp, current + cubeOffsets[5], log2Size).x,
            get_voxel<T>(hp, current + cubeOffsets[6], log2Size).x,
            get_voxel<T>(hp, current + cubeOffsets[7], log2Size).x
    };

    int acc = current.w + neighbors[0];
    bool cmp[8];
    cmp[0] = acc <= target;
    acc += neighbors[1];
    cmp[1] = acc <= target;
    acc += neighbors[2];
    cmp[2] = acc <= target;
    acc += neighbors[3];
    cmp[3] = acc <= target;
    acc += neighbors[4];
    cmp[4] = acc <= target;
    acc += neighbors[5];
    cmp[5] = acc <= target;
    acc += neighbors[6];
    cmp[6] = acc <= target;
    cmp[7] = 0;

    unsigned int sum = (cmp[0]+cmp[1]+cmp[2]+cmp[3]+cmp[4]+cmp[5]+cmp[6]+cmp[7]);
    uint4 offset = cubeOffsets[sum];
    current += offset;
    current.x = current.x*2;
    current.y = current.y*2;
    current.z = current.z*2;
    current.w = current.w +
        cmp[0]*neighbors[0] + 
        cmp[1]*neighbors[1] + 
        cmp[2]*neighbors[2] + 
        cmp[3]*neighbors[3] + 
        cmp[4]*neighbors[4] + 
        cmp[5]*neighbors[5] + 
        cmp[6]*neighbors[6] + 
        cmp[7]*neighbors[7];
    return current;
}

// each thread computes a triangle. target is computed from the position on the
// grid and position of the thread in the block. target is the triangle number
// we wish to create. we walk the histopyramid down using target and find the
// voxel for which we create the triangle.
__global__ void traverseHP(
        float3 * VBOBuffer,
        int isolevel,
        int sum,
        int log2Size,
        unsigned int size,
        unsigned int log2CubeWidth
        ) {
    unsigned int target = (blockIdx.y * gridDim.x + blockIdx.x) << log2CubeWidth;
    target = (target + threadIdx.z) << log2CubeWidth;
    target = (target + threadIdx.y) << log2CubeWidth;
    target += threadIdx.x;
    if(target >= sum)
        return;

    // walk down the histopyramid
    uint4 cubePosition = {0,0,0,0}; // x,y,z,sum
    if (size > 512)
        cubePosition = scanHPLevel<int1>(target, levels[9], cubePosition, log2Size-9);
    
    if (size > 256)
        cubePosition = scanHPLevel<int1>(target, levels[8], cubePosition, log2Size-8);
    
    if (size > 128)
        cubePosition = scanHPLevel<int1>(target, levels[7], cubePosition, log2Size-7);

    if (size > 64)
        cubePosition = scanHPLevel<int1>(target, levels[6], cubePosition, log2Size-6);
    
    if (size > 32)
        cubePosition = scanHPLevel<int1>(target, levels[5], cubePosition, log2Size-5);

    if (size > 16)
        cubePosition = scanHPLevel<short1>(target, levels[4], cubePosition, log2Size-4);

    if (size > 8)
        cubePosition = scanHPLevel<short1>(target, levels[3], cubePosition, log2Size-3);
    cubePosition = scanHPLevel<short1>(target, levels[2], cubePosition, log2Size-2);
    cubePosition = scanHPLevel<char1>(target, levels[1], cubePosition, log2Size-1);
    cubePosition = scanHPLevel<uchar4>(target, levels[0], cubePosition, log2Size);
    // revert last multiplikation in scanHPLevel
    cubePosition.x = cubePosition.x / 2;
    cubePosition.y = cubePosition.y / 2;
    cubePosition.z = cubePosition.z / 2;

    char vertexNr = 0;
    const uchar4 cubeData = get_voxel<uchar4>(levels[0], cubePosition, log2Size);

    // seems to compute one triangle
    // max 5 triangles
    for(int i = (target-cubePosition.w)*3; i < (target-cubePosition.w+1)*3; i++) { // for each vertex in triangle
        const unsigned char edge = triTable[cubeData.y*16 + i];
        const int3 point0 = make_int3(cubePosition.x + offsets3[edge*6], cubePosition.y + offsets3[edge*6+1], cubePosition.z + offsets3[edge*6+2]);
        const int3 point1 = make_int3(cubePosition.x + offsets3[edge*6+3], cubePosition.y + offsets3[edge*6+4], cubePosition.z + offsets3[edge*6+5]);

	// forwardDifferences are needed to compute the normal
        const float3 forwardDifference0 = make_float3(
                (float)(-get_voxel<uchar4>(levels[0], make_uint4(point0.x+1, point0.y, point0.z, 0), log2Size).z + get_voxel<uchar4>(levels[0], make_uint4(point0.x-1, point0.y, point0.z, 0), log2Size).z), 
                (float)(-get_voxel<uchar4>(levels[0], make_uint4(point0.x, point0.y+1, point0.z, 0), log2Size).z + get_voxel<uchar4>(levels[0], make_uint4(point0.x, point0.y-1, point0.z, 0), log2Size).z), 
                (float)(-get_voxel<uchar4>(levels[0], make_uint4(point0.x, point0.y, point0.z+1, 0), log2Size).z + get_voxel<uchar4>(levels[0], make_uint4(point0.x, point0.y, point0.z-1, 0), log2Size).z) 
            );

        const float3 forwardDifference1 = make_float3(
                (float)(-get_voxel<uchar4>(levels[0], make_uint4(point1.x+1, point1.y, point1.z, 0), log2Size).z + get_voxel<uchar4>(levels[0], make_uint4(point1.x-1, point1.y, point1.z, 0), log2Size).z), 
                (float)(-get_voxel<uchar4>(levels[0], make_uint4(point1.x, point1.y+1, point1.z, 0), log2Size).z + get_voxel<uchar4>(levels[0], make_uint4(point1.x, point1.y-1, point1.z, 0), log2Size).z), 
                (float)(-get_voxel<uchar4>(levels[0], make_uint4(point1.x, point1.y, point1.z+1, 0), log2Size).z + get_voxel<uchar4>(levels[0], make_uint4(point1.x, point1.y, point1.z-1, 0), log2Size).z) 
            );

        const int value0 = get_voxel<uchar4>(levels[0], make_uint4(point0.x, point0.y, point0.z, 0), log2Size).z;

        const float diff = (isolevel-value0) / (float)(get_voxel<uchar4>(levels[0], make_uint4(point1.x, point1.y, point1.z, 0), log2Size).z - value0);
        
        const float3 vertex = mix(make_float3(point0.x, point0.y, point0.z), make_float3(point1.x, point1.y, point1.z), diff);

        const float3 normal = mix(forwardDifference0, forwardDifference1, diff);

        // Store vertex and normal in VBO
        VBOBuffer[target*6 + vertexNr*2] = vertex;
        VBOBuffer[target*6 + vertexNr*2 + 1] = normal;

        ++vertexNr;
    }
}

#endif // gpu_mc_kernel_h__
