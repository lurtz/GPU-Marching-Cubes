#ifndef gpu_mc_h__
#define gpu_mc_h__

#define DEBUG
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>

// loads voxeldata onto the gpu
// voxels needs to be cubic with sidelength of a power of 2
void setupCuda(unsigned char * voxels, unsigned int size, GLuint vbo);

// classifies each voxel and calculates the number of triangles needed for this
// voxel
void updateScalarField();

// calculates the total number of triangles needed
void histoPyramidConstruction();

// creates the VBO
void histoPyramidTraversal();

#ifdef DEBUG
bool testUpdateScalarField(unsigned char * voxels);
bool testHistoPyramidConstruction();
bool testHistoPyramidTraversal();
#endif // DEBUG

#endif // gpu_mc_h__
