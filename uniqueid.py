#!/usr/bin/env python

import array
import math
from collections import namedtuple

cudavector = namedtuple('cudavector', ['x', 'y', 'z'])
cube_width = 4
log2_cube_width = 2

def calcId(blockIdx, threadIdx, blockDim, log2CubeWidth):
    target = (blockIdx.x * blockDim.y + blockIdx.y) << log2CubeWidth
    target = (target + threadIdx.z) << log2CubeWidth
    target = (target + threadIdx.y) << log2CubeWidth
    target += threadIdx.x
    return target

def createIds(number):
    ids = array.array('I', range(0, number))

    number_of_blocks = number/cube_width/cube_width/cube_width;
    grid_dim_x = int(math.floor(math.sqrt(number_of_blocks)));
    grid_dim_y = int(math.ceil(number_of_blocks/grid_dim_x));
    grid = cudavector(grid_dim_x, grid_dim_y, 1);
    i = 0
    for gx in range(0, grid_dim_x):
        for gy in range(0, grid_dim_y):
            grid = cudavector(gx, gy, 1)
            for z in range(0, cube_width):
                for y in range(0, cube_width):
                    for x in range(0, cube_width):
                        ids[i] = calcId(grid, cudavector(x, y, z), grid, log2_cube_width)
                        i += 1

    return ids

def testIds(number):
    ids = createIds(number)
    return len(set(ids))

numbers = 100
if numbers == testIds(numbers):
  print "alles ok\n"
else:
  print "es gibt duplikate\n"
