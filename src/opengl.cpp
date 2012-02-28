#include <GL/glew.h>
#include <GL/glut.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include "opengl.h"

typedef struct {
    int x,y,z;
} Size;

typedef struct {
    float x,y,z;
} Sizef;

// Define some globals
GLuint VBO_ID = 0;
int windowWidth, windowHeight;

Sizef scalingFactor;
Sizef translation;

float camX, camY, camZ = 4.0f; //X, Y, and Z
float lastx, lasty, xrot, yrot, xrotrad, yrotrad; //Last pos and rotation
float speed = 0.1f; //Movement speed

void mouseMovement(int x, int y) {
    int cx = windowWidth/2;
    int cy = windowHeight/2;

    if(x == cx && y == cy){ //The if cursor is in the middle
        return;
    }

    int diffx=x-cx; //check the difference between the current x and the last x position
    int diffy=y-cy; //check the difference between the current y and the last y position
    xrot += (float)diffy/2; //set the xrot to xrot with the addition of the difference in the y position
    yrot += (float)diffx/2;// set the xrot to yrot with the addition of the difference in the x position
    glutWarpPointer(cx, cy); //Bring the cursor to the middle
}

void renderBitmapString(float x, float y, float z, void *font, char *string) {
    glRasterPos3f(x, y,z);
    for(char * c = string; *c != '\0'; c++) {
        glutBitmapCharacter(font, *c);
    }
}

int frame = 0;
int timebase = 0;
std::stringstream s;
int previousTime = 0;

void drawFPSCounter(int sum) {
    frame++;

    int time = glutGet(GLUT_ELAPSED_TIME);
    if (time - timebase > 1000) { // 1 times per second
        s.str("");
        s << "Marching Cubes - Triangles: " << sum
          << " FPS: " << frame*1000.0/(time-timebase)
          << " Speed: " << static_cast<int>(time - previousTime) << " ms"
          << std::endl;
        timebase = time;
        frame = 0;
    }

    previousTime = time;
    glutSetWindowTitle(s.str().c_str());
}

void idle() {
    glutPostRedisplay();
}

void reshape(int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, width, height);
    gluPerspective(45.0f, (GLfloat)width/(GLfloat)height, 0.5f, 10000.0f);
}

void renderScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//    histoPyramidConstruction();

    // Read top of histoPyramid an use this size to allocate VBO below
//      int sum[8] = {0,0,0,0,0,0,0,0};
//    queue.enqueueReadImage(images[images.size()-1], CL_FALSE, origin, region, 0, 0, sum);

//      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
//      queue.finish();
//      int totalSum = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7] ;

//      if (totalSum == 0) {
//          std::cout << "HistoPyramid result is 0" << std::endl;
//          return;
//      }

        // 128 MB
        //if(totalSum >= 1864135) // Need to split into several VBO's to support larger structures
        //      isolevel_up = true;

        // Create new VBO
//        glGenBuffers(1, &VBO_ID);
//        glBindBuffer(GL_ARRAY_BUFFER, VBO_ID);
//        glBufferData(GL_ARRAY_BUFFER, totalSum*18*sizeof(cl_float), NULL, GL_STATIC_DRAW);
//        //std::cout << "VBO using: " << sum[0]*18*sizeof(cl_float) / (1024*1024) << " M bytes" << std::endl;
//        glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Traverse the histoPyramid and fill VBO
//        histoPyramidTraversal(totalSum);

    // Render VBO
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glRotatef(270.0f, 1.0f, 0.0f, 0.0f);  
//    drawFPSCounter(totalSum);

    glTranslatef(-camX, -camY, -camZ);

    glRotatef(xrot,1.0,0.0,0.0);
    glRotatef(yrot,0.0, 1.0, 0.0);

    glPushMatrix();
    glColor3f(1.0f, 1.0f, 1.0f);
    glScalef(scalingFactor.x, scalingFactor.y, scalingFactor.z);
    glTranslatef(translation.x, translation.y, translation.z);

    glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
    // Normal Buffer
//    glBindBuffer(GL_ARRAY_BUFFER, VBO_ID);
//    glEnableClientState(GL_VERTEX_ARRAY);
//    glEnableClientState(GL_NORMAL_ARRAY);

//    glVertexPointer(3, GL_FLOAT, 24, BUFFER_OFFSET(0));
//    glNormalPointer(GL_FLOAT, 24, BUFFER_OFFSET(12));

//      queue.finish();
      //glWaitSync(traversalSync, 0, GL_TIMEOUT_IGNORED);
//    glDrawArrays(GL_TRIANGLES, 0, totalSum*3);

    glutSolidSphere(70.0, 100, 100);

    // Release buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glPopMatrix();
    glutSwapBuffers();
//    glDeleteBuffers(1, &VBO_ID);
}

void run() {
    glutMainLoop();
}

void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case '+':
//            isolevel ++;
            break;
        case '-':
//            isolevel --;
            break;
        //WASD movement
        case 'w':
            camZ -= 0.1f;
        break;
        case 's':
            camZ += 0.1f;
        break;
        case 'a':
            camX -= 0.1f;
            break;
        case 'd':
            camX += 0.1f;
        break;
        case 27:
            //TODO some clean up
            exit(0);
        break;
        }
}

void setupOpenGL(int * argc, char ** argv, int size, int sizeX, int sizeY, int sizeZ, float spacingX, float spacingY, float spacingZ) {
/* Initialize GLUT */
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH),glutGet(GLUT_SCREEN_HEIGHT));
    glutCreateWindow("GPU Marching Cubes");
    glutFullScreen();
    glutDisplayFunc(renderScene);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(mouseMovement);

    glewInit();
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);

    // Set material properties which will be assigned by glColor
    GLfloat color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    GLfloat specReflection[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specReflection);
    GLfloat shininess[] = { 16.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);

    // Create light components
    GLfloat ambientLight[] = { 0.3f, 0.3f, 0.3f, 1.0f };
    GLfloat diffuseLight[] = { 0.7f, 0.7f, 0.7f, 1.0f };
    GLfloat specularLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat position[] = { -0.0f, 4.0f, 1.0f, 1.0f };

    // Assign created components to GL_LIGHT0
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
    glLightfv(GL_LIGHT0, GL_POSITION, position);

/*
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = 2;
        region[1] = 2;
        region[2] = 2;
*/
    scalingFactor.x = spacingX*1.5f/size;
    scalingFactor.y = spacingY*1.5f/size;
    scalingFactor.z = spacingZ*1.5f/size;

    translation.x = (float)sizeX/2.0f;
    translation.y = -(float)sizeY/2.0f;
    translation.z = -(float)sizeZ/2.0f;

    glGenBuffers(1, &VBO_ID);
}

GLuint getVBO() {
    return VBO_ID;
}
