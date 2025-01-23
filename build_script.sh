#!/bin/bash

export PATH=/home/user/cmake_3_30_6_version/cmake-3.30.6-build/bin:$PATH
cmake -DUSE_OPENMP=0 -DUSE_MPI=0 -B build_serial -S .
