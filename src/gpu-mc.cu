#include "gpu-mc.h"
#include "gpu-mc-kernel.h"
#include <utility>
#include <vector>

const unsigned int CUBESIZE = 8;
const unsigned int LOG2CUBESIZE = 3;
const unsigned int CUBESIZEHP = 2;
const unsigned int LOG2CUBESIZEHP = 1;
unsigned int SIZE;
unsigned int rawMemSize;
unsigned char * rawDataPtr;

// first level has char4 as datatype, which contains: (number of triangles, cube index, value of first cube element, 0)
// first to second level contain volumes with unsigned char as elements
// third to fifth (including) level contain unsigned short as elements
// sixth level and more uses int
std::vector<std::pair<unsigned int, void*> > images_size_pointer;

int isolevel = 50;

void setupCuda(unsigned char * voxels, unsigned int size) {
    SIZE = size;

    // Create images for the HistogramPyramid
    unsigned int bufferSize = SIZE;
    void * tmpDataPtr = NULL;
    // Make the two first buffers use INT8
    cudaMalloc((void **) &tmpDataPtr, bufferSize * bufferSize * bufferSize * sizeof(uchar4));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    bufferSize /= 2;
    cudaMalloc((void **) &tmpDataPtr, bufferSize * bufferSize * bufferSize * sizeof(unsigned char));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    bufferSize /= 2;
    // And the third, fourth and fifth INT16
    cudaMalloc((void **) &tmpDataPtr, bufferSize * bufferSize * bufferSize * sizeof(unsigned short));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    bufferSize /= 2;
    cudaMalloc((void **) &tmpDataPtr, bufferSize * bufferSize * bufferSize * sizeof(unsigned short));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    bufferSize /= 2;
    cudaMalloc((void **) &tmpDataPtr, bufferSize * bufferSize * bufferSize * sizeof(unsigned short));
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    bufferSize /= 2;
    // The rest will use INT32
    for(int i = 5; i < (log2((float)SIZE)); i++) {
        if(bufferSize == 1)
            bufferSize = 2; // Image cant be 1x1x1
        cudaMalloc((void **) &tmpDataPtr, bufferSize * bufferSize * bufferSize * sizeof(unsigned int));
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
        bufferSize /= 2;
    }

    // Transfer dataset to device
    rawMemSize = SIZE*SIZE*SIZE*sizeof(unsigned char);
    cudaMalloc((void **) &rawDataPtr, rawMemSize);
    cudaMemcpy(rawDataPtr, voxels, rawMemSize, cudaMemcpyHostToDevice);
    delete[] voxels;
}

int log2(unsigned int val) {
    int log2Val = 0;
    while (val > 1) {
      val /= 2; log2Val++;
    }
    return log2Val;
}

void updateScalarField() {
    unsigned int _size = images_size_pointer[0].first;
    dim3 block(CUBESIZE, CUBESIZE, CUBESIZE);
    dim3 grid((_size / CUBESIZE) * (_size / CUBESIZE), _size / CUBESIZE, 1);
    int log2GridSize = log2(_size / CUBESIZE);
    kernelClassifyCubes<<<grid , block>>>((uchar4 *)(images_size_pointer[0].second), rawDataPtr, isolevel, log2GridSize, _size/CUBESIZE-1, LOG2CUBESIZE, _size);
}

void histoPyramidConstruction() {
    updateScalarField();

    unsigned int i = 0;
    dim3 block(CUBESIZEHP, CUBESIZEHP, CUBESIZEHP);
    
    if (i < log2((float)SIZE)-1) {
        unsigned int _size = images_size_pointer[i+1].first;
        dim3 grid((_size / CUBESIZEHP) * (_size / CUBESIZEHP), _size / CUBESIZEHP, 1);
        int log2GridSize = log2(_size / CUBESIZEHP);
        kernelConstructHPLevel2<<<grid, block>>>((uchar4 *)(images_size_pointer[i].second) , (unsigned char *)(images_size_pointer[i+1].second), log2GridSize, _size/CUBESIZEHP-1, LOG2CUBESIZEHP, _size); 
        i++;
    }
    
    if (i < log2((float)SIZE)-1) {
        unsigned int _size = images_size_pointer[i+1].first;
        dim3 grid((_size / CUBESIZEHP) * (_size / CUBESIZEHP), _size / CUBESIZEHP, 1);
        int log2GridSize = log2(_size / CUBESIZEHP);
        kernelConstructHPLevel3<<<grid, block>>>((unsigned char *)(images_size_pointer[i].second) , (unsigned short *)(images_size_pointer[i+1].second), log2GridSize, _size/CUBESIZEHP-1, LOG2CUBESIZEHP, _size); 
        i++;
    }
    
    if (i < log2((float)SIZE)-1) {
        unsigned int _size = images_size_pointer[i+1].first;
        dim3 grid((_size / CUBESIZEHP) * (_size / CUBESIZEHP), _size / CUBESIZEHP, 1);
        int log2GridSize = log2(_size / CUBESIZEHP);
        kernelConstructHPLevel45<<<grid, block>>>((unsigned short *)(images_size_pointer[i].second) , (unsigned short *)(images_size_pointer[i+1].second), log2GridSize, _size/CUBESIZEHP-1, LOG2CUBESIZEHP, _size); 
        i++;
    }
    
    if (i < log2((float)SIZE)-1) {
        unsigned int _size = images_size_pointer[i+1].first;
        dim3 grid((_size / CUBESIZEHP) * (_size / CUBESIZEHP), _size / CUBESIZEHP, 1);
        int log2GridSize = log2(_size / CUBESIZEHP);
        kernelConstructHPLevel45<<<grid, block>>>((unsigned short *)(images_size_pointer[i].second) , (unsigned short *)(images_size_pointer[i+1].second), log2GridSize, _size/CUBESIZEHP-1, LOG2CUBESIZEHP, _size); 
        i++;
    }
    
    if (i < log2((float)SIZE)-1) {
        unsigned int _size = images_size_pointer[i+1].first;
        dim3 grid((_size / CUBESIZEHP) * (_size / CUBESIZEHP), _size / CUBESIZEHP, 1);
        int log2GridSize = log2(_size / CUBESIZEHP);
        kernelConstructHPLevel6<<<grid, block>>>((unsigned short *)(images_size_pointer[i].second) , (unsigned int *)(images_size_pointer[i+1].second), log2GridSize, _size/CUBESIZEHP-1, LOG2CUBESIZEHP, _size); 
        i++;
    }
    
    // Run level 7 to top level
    for(; i < log2((float)SIZE)-1; i++) {
        unsigned int _size = images_size_pointer[i+1].first;
        dim3 grid((_size / CUBESIZEHP) * (_size / CUBESIZEHP), _size / CUBESIZEHP, 1);
        int log2GridSize = log2(_size / CUBESIZEHP);
        kernelConstructHPLevel<<<grid, block>>>((unsigned int *)(images_size_pointer[i].second) , (unsigned int *)(images_size_pointer[i+1].second), log2GridSize, _size/CUBESIZEHP-1, LOG2CUBESIZEHP, _size); 
    }
}

