/* see LICENSE file for copyright and license details */

#ifndef __OPENGL_H__
#define __OPENGL_H__

#include <GL/gl.h>

void setupOpenGL(int * argc, char ** argv, int size, int sizeX, int sizeY, int sizeZ, float spacingX, float spacingY, float spacingZ);
void run();
GLuint getVBO();

#endif // __OPENGL_H__
