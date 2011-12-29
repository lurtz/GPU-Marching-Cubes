#include <device_functions.h>

__device__ __constant__ uint4 cubeOffsets[8] = {
		{0, 0, 0, 0},
		{1, 0, 0, 0},
		{0, 0, 1, 0},
		{1, 0, 1, 0},
		{0, 1, 0, 0},
		{1, 1, 0, 0},
		{0, 1, 1, 0},
		{1, 1, 1, 0},
	}; 

__device__ __constant__ unsigned char nrOfTriangles[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5, 2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3, 4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0};

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

__device__ unsigned int getId(uint4 pos, int log2BlockWidth, int log2CubeWidth) {
    int log2Sum = log2BlockWidth + log2CubeWidth;
    return (((pos.z << log2Sum) + pos.y) << log2Sum) + pos.x;
}

__device__ int4 getPosFromId2(unsigned int id, int log2BlockWidth, int log2CubeWidth) {
    int log2Sum = log2BlockWidth + log2CubeWidth;
    // TODO untested, maybe a big fail
    // assumes that the bits of x,y,z are seperated in id 
    // the width of the volume must be a multiple of 2
    // TODO if this works, eliminate loop by lookuptable
    int pattern = 0x00;
    for (unsigned int i = 0; i < log2Sum; i++)
        pattern |= 1 << i;
    int x = id & pattern;
    id = id >> log2Sum;
    int y = id & pattern;
    id = id >> log2Sum;
    int z = id & pattern;
    return make_int4(x, y, z, 0);
}

__device__ int4 getPosFromId(unsigned int id, int size) {
  int z = id / (size * size);
  id = id % (size * size);
  int y = id / size;
  int x = id % size;
  return make_int4(x, y, z, 0);
}

// addition
inline __host__ __device__ uchar4 operator+(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// addition
inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void kernelClassifyCubes(cudaPitchedPtr histoPyramid, unsigned char * rawData, int isolevel, int log2BlockWidth, int mask, int log2CubeWidth, unsigned int volumeSize) {
    // 2d grid
    // 3d kernel
    uint4 pos = getPosFromGrid(blockIdx, threadIdx, log2BlockWidth, mask, log2CubeWidth);
    
    unsigned int id = getId(pos, log2BlockWidth, log2CubeWidth);

    unsigned char first = rawData[getId(pos, log2BlockWidth, log2CubeWidth)];

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
    char* devPtr = (char*)histoPyramid.ptr;
    size_t pitch = histoPyramid.pitch;
    size_t slicePitch = pitch << (log2BlockWidth + log2CubeWidth);
    char* slice = devPtr + pos.z * slicePitch;
    uchar4* row = (uchar4*)(slice + pos.y * pitch);
    row[pos.x] = ret_val;
}

// multiplication
inline __host__ __device__ uint4 operator*(uint4 a, int b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// possibly a huge fail
__global__ void kernelConstructHPLevel(cudaPitchedPtr readHistoPyramid, cudaPitchedPtr writeHistoPyramid, int log2BlockWidth, int mask, int log2CubeWidth, unsigned char readElementSize, unsigned char writeElementSize, bool vector) {    
    uint4 writePos = getPosFromGrid(blockIdx, threadIdx, log2BlockWidth, mask, log2CubeWidth);
    uint4 readPos = writePos*2;

    char* devPtr = (char*)readHistoPyramid.ptr;
    size_t pitch = readHistoPyramid.pitch;
    size_t slicePitch = pitch << (log2BlockWidth + log2CubeWidth);
    int writeValue = 0;
    for (unsigned int i = 0; i < 8; i++) {
        uint4 tmpPos = readPos+cubeOffsets[i];
        char* slice = devPtr + tmpPos.z * slicePitch;
        char* row = slice + tmpPos.y * pitch;
        // first level has uchar4 as datatype
        if (!vector)
            writeValue += row[tmpPos.x*readElementSize];
        else
            writeValue += ((uchar4*)row)[tmpPos.x].x;
    }

    // level2 has half side length
    devPtr = (char*)writeHistoPyramid.ptr;
    pitch = writeHistoPyramid.pitch;
    slicePitch = pitch << (log2BlockWidth-1 + log2CubeWidth);
    char* slice = devPtr + writePos.z * slicePitch;
    char* row = slice + writePos.y * pitch;
    // hopefully this will write char, short and int
    int* writePosPtr = (int*)(row + writePos.x*writeElementSize);
    *writePosPtr = writeValue;
}

