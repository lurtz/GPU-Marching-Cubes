#Include the FindCUDA script
INCLUDE(FindCUDA)

#The rule to build the executable program 
CUDA_ADD_EXECUTABLE(mc main.cpp rawUtilities.cpp gpu-mc.cu opengl.cpp)
TARGET_LINK_LIBRARIES(mc
    ${OpenGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GL_LIBRARY}
    ${OPENGL_LIBRARIES}
    )
#TARGET_LINK_LIBRARIES(pba ${OpenCV_LIBS})
