/* see LICENSE file for copyright and license details */

#ifndef RAW_UTILITIES
#define RAW_UTILITIES

#include <vector>
#include <string>

unsigned char * readRawFile(char const * const, int, int, int, int, int, int);
unsigned char * readRawFile(std::string, const std::vector<unsigned int>&, int, int, int);
int prepareDataset(unsigned char ** voxels, int sizeX, int sizeY, int sizeZ);
int prepareDataset(unsigned char ** voxels, const std::vector<unsigned int>& dim);
inline int toid(int x, int y, int z, int size);
inline int toid(int x, int y, int z, int sizeX, int sizeY, int sizeZ);
inline int toid(int x, int y, int z, const std::vector<unsigned int>& dim);
std::vector<unsigned int> read_dimensions(const std::string file_dim);
unsigned char * create_voxel_sphere(const std::vector<unsigned int>& dim);

#endif
