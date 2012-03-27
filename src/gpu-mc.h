/* see LICENSE file for copyright and license details */

#ifndef gpu_mc_h__
#define gpu_mc_h__

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>

// loads voxeldata onto the gpu
// voxels needs to be cubic with sidelength of a power of 2
// if vbo is zero the kernels will only run to test the speed and exit
void setupCuda(unsigned char * voxels, unsigned int size, GLuint vbo);

// runs the marching cube and writes the triangle mesh to the vbo supplied at 
// setup
int marching_cube(int _isolevel);

#endif // gpu_mc_h__
