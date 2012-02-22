// loads voxeldata onto the gpu
// voxels needs to be cubic with sidelength of a power of 2
void setupCuda(unsigned char * voxels, unsigned int size);

// classifies each voxel and calculates the number of triangles needed for this
// voxel
void updateScalarField();
bool testUpdateScalarField(unsigned char * voxels);

// calculates the total number of triangles needed
void histoPyramidConstruction();
