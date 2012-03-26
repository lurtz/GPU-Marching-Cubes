/* see LICENSE file for copyright and license details */

#ifndef RAW_UTILITIES
#define RAW_UTILITIES

#include <vector>
#include <string>

// read voxeldata from a file, given its dimensions
unsigned char * readRawFile(char const * const, int, int, int, int, int, int);
unsigned char * readRawFile(std::string, const std::vector<unsigned int>&, int, int, int);
// makes sure the voxelvolume has sidelengths of a power of 2
int prepareDataset(unsigned char ** voxels, int sizeX, int sizeY, int sizeZ);
int prepareDataset(unsigned char ** voxels, const std::vector<unsigned int>& dim);
// creates a 1D index of a 3D position
inline int toid(int x, int y, int z, int size);
inline int toid(int x, int y, int z, int sizeX, int sizeY, int sizeZ);
inline int toid(int x, int y, int z, const std::vector<unsigned int>& dim);
// read dimensions of voxeldata from a file
std::vector<unsigned int> read_dimensions(const std::string file_dim);
// create a sphere
unsigned char * create_voxel_sphere(const std::vector<unsigned int>& dim);

#endif
