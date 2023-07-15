# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
