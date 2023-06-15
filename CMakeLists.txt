cmake_minimum_required(VERSION 3.10)

project(tcpgpudmarxd VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
find_package(CUDAToolkit)
find_package(CUDA REQUIRED)

find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

ADD_SUBDIRECTORY(proto)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(test)
