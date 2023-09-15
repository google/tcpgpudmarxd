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

enable_language(CUDA)
find_package(CUDAToolkit)
find_package(CUDA REQUIRED)
find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)


add_executable(tcpgpudmarxd tcpgpudmarxd.cu)
target_include_directories(tcpgpudmarxd PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(tcpgpudmarxd PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(tcpgpudmarxd cuda_common)
target_link_libraries(tcpgpudmarxd gpu_rxq_configurator)
target_link_libraries(tcpgpudmarxd nic_configurator)
target_link_libraries(tcpgpudmarxd rx_rule_manager)
target_link_libraries(tcpgpudmarxd gpu_page_exporter_factory)
target_link_libraries(tcpgpudmarxd tcpgpudmarxd_proto)
target_link_libraries(tcpgpudmarxd google_status_proto)
target_link_libraries(tcpgpudmarxd ${Protobuf_LIBRARIES})
target_link_libraries(tcpgpudmarxd absl::absl_check absl::flags absl::flags_parse absl::log absl::str_format absl::debugging absl::debugging_internal absl::failure_signal_handler)


install(TARGETS tcpgpudmarxd DESTINATION "${app_dest}")