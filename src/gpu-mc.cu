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

unsigned int sum_of_triangles = 0;

// first level has char4 as datatype, which contains: (number of triangles, cube index, value of first cube element, 0)
// first to second level contain volumes with unsigned char as elements
// third to fifth (including) level contain unsigned short as elements
// sixth level and more uses int
std::vector<std::pair<cudaExtent, cudaPitchedPtr> > images_size_pointer;

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

    bufferSize.width = bufferSize.depth/2 * sizeof(uchar1);
    bufferSize.height = bufferSize.depth/2;
    bufferSize.depth = bufferSize.depth/2;
    cudaMalloc3D(&tmpDataPtr, bufferSize);
    images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));

    // And the third, fourth and fifth INT16
    for (unsigned int i = 0; i < 3; i++) {
        bufferSize.width = bufferSize.depth/2 * sizeof(ushort1);
        bufferSize.height = bufferSize.depth/2;
        bufferSize.depth = bufferSize.depth/2;
        cudaMalloc3D(&tmpDataPtr, bufferSize);
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    }

    // The rest will use INT32
    for(int i = 5; i < (log2((float)SIZE)); i++) {
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
        images_size_pointer.push_back(std::make_pair(bufferSize, tmpDataPtr));
    }

    // Transfer dataset to device
    rawMemSize = SIZE*SIZE*SIZE*sizeof(unsigned char);
    cudaMalloc((void **) &rawDataPtr, rawMemSize);
    cudaMemcpy(rawDataPtr, voxels, rawMemSize, cudaMemcpyHostToDevice);
//    delete[] voxels;
}

template<typename T>
T log2(T val) {
    T log2Val = 0;
    while (val > 1) {
      val /= 2; log2Val++;
    }
    return log2Val;
}

void updateScalarField() {
    cudaExtent _size = images_size_pointer[0].first;
    dim3 block(CUBESIZE, CUBESIZE, CUBESIZE);
    dim3 grid((_size.depth / CUBESIZE) * (_size.depth / CUBESIZE), _size.depth / CUBESIZE, 1);
    int log2GridSize = log2(_size.depth / CUBESIZE);
    kernelClassifyCubes<<<grid , block>>>(images_size_pointer[0].second, rawDataPtr, isolevel, log2GridSize, _size.depth/CUBESIZE-1, LOG2CUBESIZE, _size.depth);
}

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

char bit2Offset[] = {0, 1, 3, 2, 4, 5, 7, 6};
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

bool testUpdateScalarField(unsigned char * voxels) {
    // get level0 data from gpu
    uchar4 * lvl0_data = new uchar4[SIZE*SIZE*SIZE];
    cudaPitchedPtr h_pitched_ptr = make_cudaPitchedPtr(lvl0_data, SIZE*sizeof(uchar4), SIZE, SIZE);
    struct cudaMemcpy3DParms parms = {0};
    parms.srcPtr = images_size_pointer[0].second;
    parms.dstPtr = h_pitched_ptr;
    parms.extent = images_size_pointer[0].first;
    parms.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&parms);

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

void histoPyramidConstruction() {
    // first level
    updateScalarField();

    dim3 block(CUBESIZEHP, CUBESIZEHP, CUBESIZEHP);
    
    unsigned int i = 0;
    // second level
    if (i < log2((float)SIZE)-1) {
        cudaExtent _size = images_size_pointer[i+1].first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        kernelConstructHPLevel<uchar4, uchar1><<<grid, block>>>(images_size_pointer[i].second , images_size_pointer[i+1].second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
    }
    i++;

    // third level
    if (i < log2((float)SIZE)-1) {
        cudaExtent _size = images_size_pointer[i+1].first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        kernelConstructHPLevel<uchar1, ushort1><<<grid, block>>>(images_size_pointer[i].second , images_size_pointer[i+1].second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
    }
    i++;

    // fourth, fifth level
    for (unsigned int j = 0; i < log2((float)SIZE)-1 && j < 2; i++, j++) {
        cudaExtent _size = images_size_pointer[i+1].first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        kernelConstructHPLevel<ushort1, ushort1><<<grid, block>>>(images_size_pointer[i].second , images_size_pointer[i+1].second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
    }

    // sixth level
    if (i < log2((float)SIZE)-1) {
        cudaExtent _size = images_size_pointer[i+1].first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        kernelConstructHPLevel<ushort4, uint1><<<grid, block>>>(images_size_pointer[i].second , images_size_pointer[i+1].second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
    }
    i++;

    // all other levels
    for (; i < log2((float)SIZE)-1; i++) {
        cudaExtent _size = images_size_pointer[i+1].first;
        dim3 grid((_size.depth / CUBESIZEHP) * (_size.depth / CUBESIZEHP), _size.depth / CUBESIZEHP, 1);
        int log2GridSize = log2(_size.depth / CUBESIZEHP);
        kernelConstructHPLevel<uint1, uint1><<<grid, block>>>(images_size_pointer[i].second , images_size_pointer[i+1].second, log2GridSize, _size.depth/CUBESIZEHP-1, LOG2CUBESIZEHP); 
    }
}
