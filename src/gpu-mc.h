#ifndef gpu_mc_h__
#define gpu_mc_h__

#define DEBUG

// loads voxeldata onto the gpu
// voxels needs to be cubic with sidelength of a power of 2
void setupCuda(unsigned char * voxels, unsigned int size);

// classifies each voxel and calculates the number of triangles needed for this
// voxel
void updateScalarField();
#ifdef DEBUG
bool testUpdateScalarField(unsigned char * voxels);
#endif // DEBUG

// calculates the total number of triangles needed
void histoPyramidConstruction();
#ifdef DEBUG
bool testHistoPyramidConstruction();
#endif // DEBUG

// creates the VBO
void histoPyramidTraversal();

#endif // gpu_mc_h__
