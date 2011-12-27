#include "gpu-mc.h"
#include <utility>
#include <vector>

unsigned int SIZE;
unsigned int rawMemSize;
int * rawDataPtr;

// first two level contain volumes with unsigned char as elements
// third to fifth (including) level contain unsigned short as elements
// sixth level and more uses int
std::vector<std::pair<unsigned int, void*> > images_size_pointer;

void setupCuda(unsigned char * voxels, unsigned int size) {
    SIZE = size;

    // Create images for the HistogramPyramid
    unsigned int bufferSize = SIZE;
    int * tmpDataPtr = NULL;
    // Make the two first buffers use INT8
    cudaMalloc((void **) &tmpDataPtr, bufferSize * bufferSize * bufferSize * sizeof(unsigned char));
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
