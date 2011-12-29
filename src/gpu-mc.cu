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
std::vector<std::pair<cudaExtent, cudaPitchedPtr> > images_size_pointer;
std::vector<unsigned char> elementSizes;

int isolevel = 50;

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
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    elementSizes.push_back(sizeof(uchar4));

    bufferSize.width = bufferSize.depth/2 * sizeof(unsigned char);
    bufferSize.height = bufferSize.depth/2;
    bufferSize.depth = bufferSize.depth/2;
    cudaMalloc3D(&tmpDataPtr, bufferSize);
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    elementSizes.push_back(sizeof(unsigned char));

    // And the third, fourth and fifth INT16
    for (unsigned int i = 0; i < 3; i++) {
        bufferSize.width = bufferSize.depth/2 * sizeof(unsigned short);
        bufferSize.height = bufferSize.depth/2;
        bufferSize.depth = bufferSize.depth/2;
        cudaMalloc3D(&tmpDataPtr, bufferSize);
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
        elementSizes.push_back(sizeof(unsigned short));
    }

    // The rest will use INT32
    for(int i = 5; i < (log2((float)SIZE)); i++) {
        bufferSize.width = bufferSize.depth/2 * sizeof(unsigned int);
        bufferSize.height = bufferSize.depth/2;
        bufferSize.depth = bufferSize.depth/2;
        // Image cant be 1x1x1
        if (bufferSize.depth == 1) {
            bufferSize.width = 2 * sizeof(unsigned int);
            bufferSize.height = 2;
            bufferSize.depth = 2;
        }
        cudaMalloc3D(&tmpDataPtr, bufferSize);
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
        elementSizes.push_back(sizeof(unsigned int));
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
    cudaExtent _size = images_size_pointer[0].first;
    dim3 block(CUBESIZE, CUBESIZE, CUBESIZE);
    dim3 grid((_size.depth / CUBESIZE) * (_size.depth / CUBESIZE), _size.depth / CUBESIZE, 1);
    int log2GridSize = log2((unsigned int)_size.depth / CUBESIZE);
    kernelClassifyCubes<<<grid , block>>>(images_size_pointer[0].second, rawDataPtr, isolevel, log2GridSize, _size.depth/CUBESIZE-1, LOG2CUBESIZE, _size.depth);
}

void histoPyramidConstruction() {
    updateScalarField();

    dim3 block(CUBESIZEHP, CUBESIZEHP, CUBESIZEHP);
    
    for (unsigned int i = 0; i < log2((float)SIZE)-1; i++) {
        cudaExtent _size = images_size_pointer[i+1].first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2((unsigned int)_size.depth / CUBESIZEHP);
        kernelConstructHPLevel<<<grid, block>>>(images_size_pointer[i].second , images_size_pointer[i+1].second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP, elementSizes[i], elementSizes[i+1], i == 0); 
    }
}
