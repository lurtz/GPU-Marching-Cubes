/* see LICENSE file for copyright and license details */

#include "rawUtilities.h"
#include "gpu-mc.h"
#include "opengl.h"
#include <string>
#include <iostream>

//const bool use_opengl = false;
const bool use_opengl = true;

int main(int argc, char * argv[]) {
  int stepSizeX = 1;
  int stepSizeY = 1;
  int stepSizeZ = 1;
  float scaleX = 1;
  float scaleY = 1;
  float scaleZ = 1;
  // if a file is given, load the voxelvolume, else create a sphere
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

  // setup cuda and opengl
  int size = prepareDataset(&voxel_data_ptr, dim[0]/stepSizeX, dim[1]/stepSizeY, dim[2]/stepSizeZ);
  if (use_opengl)
    setupOpenGL(&argc,argv,size,dim[0]/stepSizeX,dim[1]/stepSizeY,dim[2]/stepSizeZ,scaleX,scaleY,scaleZ);
  setupCuda(voxel_data_ptr, size, getVBO());

  delete [] voxel_data_ptr; 

  if (use_opengl)
    run();
  else
    marching_cube(50);

  return 0;
}
