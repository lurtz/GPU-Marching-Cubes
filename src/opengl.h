/* see LICENSE file for copyright and license details */

#ifndef __OPENGL_H__
#define __OPENGL_H__

#include <GL/gl.h>

void setupOpenGL(int * argc, char ** argv, int size, int sizeX, int sizeY, int sizeZ, float spacingX, float spacingY, float spacingZ);
// starts the glutMainLoop
void run();
// returns the vbo in which will cuda write the triangle mesh
GLuint getVBO();

#endif // __OPENGL_H__
