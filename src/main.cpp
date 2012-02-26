#include "rawUtilities.h"
#include "gpu-mc.h"
#include <string>
#include <iostream>

int main(int argc, char * argv[]) {
  int stepSizeX = 1;
  int stepSizeY = 1;
  int stepSizeZ = 1;
  float scaleX = 1;
  float scaleY = 1;
  float scaleZ = 1;
  std::string file("");
  if (argc > 1)
    file = argv[1];
  std::string file_dim = file + ".dim";
  if (argc > 2)
    file_dim = argv[2];
    
  std::vector<unsigned int> dim(3);
  unsigned char * voxel_data_ptr = NULL;
  if (file != "") {
    dim = read_dimensions(file_dim);
    voxel_data_ptr = readRawFile(file, dim, stepSizeX, stepSizeY, stepSizeZ);
  }

  if (file == "" || voxel_data_ptr == NULL) {
    dim.at(0) = dim.at(1) = dim.at(2) = 100;
    voxel_data_ptr = create_voxel_sphere(dim);
  }

  int size = prepareDataset(&voxel_data_ptr, dim[0]/stepSizeX, dim[1]/stepSizeY, dim[2]/stepSizeZ);
//  setupOpenGL(&argc,argv,size,dim[0]/stepSizeX,dim[1]/stepSizeY,dim[2]/stepSizeZ,scaleX,scaleY,scaleZ);
  setupCuda(voxel_data_ptr, size);
  updateScalarField();
  bool success = testUpdateScalarField(voxel_data_ptr);

  histoPyramidConstruction();
  success &= testHistoPyramidConstruction();

  histoPyramidTraversal();
  success &= testHistoPyramidTraversal();
//  run();

  std::cout << "no segfault here" << std::endl;
  if (!success)
    std::cout << "something with the tests went wrong" << std::endl;

  delete [] voxel_data_ptr; 
  return 0;
}
