#include "rawUtilities.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cassert>
#include <limits>

int toid(int x, int y, int z, int size) {
    return toid(x, y, z, size, size, size);
}

int toid(int x, int y, int z, int sizeX, int sizeY, int sizeZ) {
    return x + y*sizeX + z*sizeX*sizeY;
}

int toid(int x, int y, int z, const std::vector<unsigned int>& dim) {
    return toid(x, y, z, dim[0], dim[1], dim[2]);
}

std::vector<unsigned int> read_dimensions(const std::string file_dim) {
    std::vector<unsigned int> dim(3);
    std::ifstream file_dim_in(file_dim.c_str());
    while (file_dim_in) {
        char token;
        unsigned int index = 3;
        file_dim_in >> token;
        if (token >= 'x' && token <= 'z') 
            index = static_cast<unsigned int>(token - 'x');
        if (index < 3 && file_dim_in) {
            file_dim_in >> token;
            assert(token == '=' && file_dim_in);
            file_dim_in >> dim[index];
        }
    }
    file_dim_in.close();
    return dim;
}

unsigned char * readRawFile(std::string filename, const std::vector<unsigned int>& dim, int stepSizeX, int stepSizeY, int stepSizeZ) {
    return readRawFile(filename.c_str(), dim[0], dim[1], dim[2], stepSizeX, stepSizeY, stepSizeZ);
}

unsigned char * readRawFile(char const * const filename, int sizeX, int sizeY, int sizeZ, int stepSizeX, int stepSizeY, int stepSizeZ) {
    // Parse the specified raw file
    int rawDataSize = sizeX*sizeY*sizeZ;
    unsigned char * rawVoxels = new unsigned char[rawDataSize];
    FILE * file = fopen(filename, "rb");
    if(file == NULL) {
        delete [] rawVoxels;
        return NULL;
    }

    fread(rawVoxels, sizeof(unsigned char), rawDataSize, file);
    if(stepSizeX == 1 && stepSizeY == 1 && stepSizeZ == 1) 
        return rawVoxels;

    unsigned char * voxels = new unsigned char[rawDataSize / ( stepSizeX*stepSizeY*stepSizeZ)];
    int i = 0;
    for(int z = 0; z < sizeZ; z += stepSizeZ) {
        for(int y = 0; y < sizeY; y += stepSizeY) {
            for(int x = 0; x < sizeX; x += stepSizeX) {
                voxels[i] = rawVoxels[toid(x, y, z, sizeX, sizeY, sizeZ)];
                i++;
            }
        }
    }

    delete [] rawVoxels;
    
    return voxels;
}

int prepareDataset(unsigned char ** voxels, const std::vector<unsigned int>& dim) {
    return prepareDataset(voxels, dim[0], dim[1], dim[2]);
}

int prepareDataset(unsigned char ** voxels, int sizeX, int sizeY, int sizeZ) {
    // If all equal and power of two exit
    if(sizeX == sizeY && sizeY == sizeZ && sizeX == pow(2, log2(sizeX)))
        return sizeX;

    // Find largest size and find closest power of two
    int largestSize = std::max(sizeX, std::max(sizeY, sizeZ));
    int size = 0;
    int i = 1;
    while(pow(2, i) < largestSize)
        i++;
    size = pow(2, i);

    // Make new voxel array of this size and fill it with zeros
    unsigned char * newVoxels = new unsigned char[size*size*size];
    for(int j = 0; j < size*size*size; j++) 
        newVoxels[j] = 0;

    // Fill the voxel array with previous data
    for(int x = 0; x < sizeX; x++) {
        for(int y = 0; y < sizeY; y++) {
            for(int z = 0; z <sizeZ; z++) {
                newVoxels[toid(x, y, z, size)] = voxels[0][toid(x, y, z, sizeX, sizeY, sizeZ)];
            }
        }
    }
    delete[] voxels[0];
    voxels[0] = newVoxels;
    return size;
}

double vector_distance(int center[], int x, int y, int z) {
  int diff_x = center[0] - x;
  int diff_y = center[1] - y;
  int diff_z = center[2] - z;
  return sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}

unsigned char * create_voxel_sphere(const std::vector<unsigned int>& dim) {
  std::cout << "Creating sphere with dimensions " << dim[0] << "x" << dim[1] << "x" << dim[2] << std::endl;
  unsigned char * voxel_data = new unsigned char[dim[0] * dim[1] * dim[2]];
  int center[] = {static_cast<int>(dim[0]/2), static_cast<int>(dim[1]/2), static_cast<int>(dim[2]/2)};
  double max_distance = vector_distance(center, static_cast<int>(dim[0]/2), static_cast<int>(dim[1]/2), 0);
  for (unsigned int x = 0; x < dim[0]; x++)
    for (unsigned int y = 0; y < dim[1]; y++)
      for (unsigned int z = 0; z < dim[2]; z++) {
        double distance = vector_distance(center, static_cast<int>(x), static_cast<int>(y), static_cast<int>(z));
        unsigned int index = toid(x, y, z, dim);
        unsigned char val = static_cast<unsigned char>(std::numeric_limits<unsigned char>::max() * std::max((1 - distance/max_distance), 0.0));
        voxel_data[index] = val;
      }
  return voxel_data;
}
