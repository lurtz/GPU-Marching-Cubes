/* see LICENSE file for copyright and license details */

#ifndef gpu_mc_h__
#define gpu_mc_h__

//#define DEBUG
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>

// loads voxeldata onto the gpu
// voxels needs to be cubic with sidelength of a power of 2
void setupCuda(unsigned char * voxels, unsigned int size, GLuint vbo);

int marching_cube(int _isolevel);

#ifdef DEBUG
bool runTests(unsigned char * voxels);
#endif // DEBUG

#endif // gpu_mc_h__
