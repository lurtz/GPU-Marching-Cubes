# The name of the project
PROJECT(Marching_Cubes)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.4)
set(CMAKE_MODULE_PATH ${Marching_Cubes_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(EXECUTABLE_OUTPUT_PATH ${Marching_Cubes_BINARY_DIR})

# Searching CUDA
FIND_PACKAGE(CUDA REQUIRED)

# Searching OpenCV
#FIND_PACKAGE(OpenCV REQUIRED)
#INCLUDE_DIRECTORIES(SYSTEM ${OpenCV_INCLUDE_DIRS})

# Compiler setup
set(CMAKE_BUILD_TYPE debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ${VALID_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Adding src directory to the project
ADD_SUBDIRECTORY(src)
