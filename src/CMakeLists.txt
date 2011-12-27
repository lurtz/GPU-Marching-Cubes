#Include the FindCUDA script
INCLUDE(FindCUDA)

#The rule to build the executable program 
CUDA_ADD_EXECUTABLE(mc main.cpp)
#TARGET_LINK_LIBRARIES(pba ${OpenCV_LIBS})
