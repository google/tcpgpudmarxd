cmake_minimum_required(VERSION 3.10)

project(tcpgpudmarxd VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(CHECK_COVERAGE "Check test coverage" OFF)

if(CHECK_COVERAGE)
    SET(CMAKE_CXX_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage")
endif(CHECK_COVERAGE)

enable_language(CUDA)
find_package(CUDAToolkit)
find_package(CUDA REQUIRED)

find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

ADD_SUBDIRECTORY(proto)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(app)

include(CTest)
ADD_SUBDIRECTORY(test)
