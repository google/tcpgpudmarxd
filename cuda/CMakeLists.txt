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
find_package(CUDAToolkit REQUIRED)
find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

# common cuda lib

set(CUDA_COMMON_SRC
	common.cu
	cuda_context_manager.cu
)

add_library(cuda_common STATIC ${CUDA_COMMON_SRC})
target_include_directories(cuda_common PRIVATE cuda)
target_include_directories(
	cuda_common
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(cuda_common PRIVATE CUDA::cuda_driver)
target_link_libraries(cuda_common PRIVATE absl::log absl::status absl::str_format)
set_target_properties(cuda_common PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# cuda ipc menhandle

set(CUDA_IPC_MEMHANDLE_SRC
	cuda_ipc_memhandle.cu
)
add_library(cuda_ipc_memhandle STATIC ${CUDA_IPC_MEMHANDLE_SRC})
target_include_directories(
	cuda_ipc_memhandle
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(cuda_ipc_memhandle PRIVATE CUDA::cuda_driver)
target_link_libraries(cuda_ipc_memhandle PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(cuda_ipc_memhandle PRIVATE cuda_common)
set_target_properties(cuda_ipc_memhandle PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# dmabuf gpu page allocator

set(DMABUF_GPU_ALLOC_SRC
	dmabuf_gpu_page_allocator.cu
)
add_library(dmabuf_gpu_page_allocator STATIC ${DMABUF_GPU_ALLOC_SRC})
target_include_directories(dmabuf_gpu_page_allocator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(dmabuf_gpu_page_allocator PRIVATE CUDA::cuda_driver)
target_link_libraries(dmabuf_gpu_page_allocator PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(dmabuf_gpu_page_allocator PRIVATE cuda_common)
set_target_properties(dmabuf_gpu_page_allocator PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# cuda ipc memhandle exporter

set(CUDA_IPC_MEMHANDLE_EXPORTER_SRC
	cuda_ipc_memhandle_exporter.cu
)
add_library(cuda_ipc_memhandle_exporter STATIC ${CUDA_IPC_MEMHANDLE_EXPORTER_SRC})
target_include_directories(cuda_ipc_memhandle_exporter PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(cuda_ipc_memhandle_exporter PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_ipc_memhandle_exporter PRIVATE CUDA::cuda_driver)
target_link_libraries(cuda_ipc_memhandle_exporter PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(cuda_ipc_memhandle_exporter PUBLIC tcpgpudmarxd_proto)
target_link_libraries(cuda_ipc_memhandle_exporter PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(cuda_ipc_memhandle_exporter PRIVATE cuda_common)
target_link_libraries(cuda_ipc_memhandle_exporter PRIVATE dmabuf_gpu_page_allocator)
target_link_libraries(cuda_ipc_memhandle_exporter PRIVATE gpu_page_exporter_interface)
target_link_libraries(cuda_ipc_memhandle_exporter PRIVATE unix_socket_server_lib)
set_target_properties(cuda_ipc_memhandle_exporter PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# cuda ipc memhandle importer
set(CUDA_IPC_MEMHANDLE_IMPORTER_SRC
	cuda_ipc_memhandle_importer.cu
)
add_library(cuda_ipc_memhandle_importer STATIC ${CUDA_IPC_MEMHANDLE_IMPORTER_SRC})
target_include_directories(cuda_ipc_memhandle_importer PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(
	cuda_ipc_memhandle_importer
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(cuda_ipc_memhandle_importer PRIVATE CUDA::cuda_driver)
target_link_libraries(cuda_ipc_memhandle_importer PRIVATE absl::log absl::absl_check absl::status absl::statusor absl::str_format)
target_link_libraries(cuda_ipc_memhandle_importer PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(cuda_ipc_memhandle_importer PRIVATE cuda_common)
target_link_libraries(cuda_ipc_memhandle_importer PRIVATE cuda_ipc_memhandle)
target_link_libraries(cuda_ipc_memhandle_importer PRIVATE unix_socket_client_lib)
target_link_libraries(cuda_ipc_memhandle_importer PRIVATE gpu_mem_importer_telemetry)
set_target_properties(cuda_ipc_memhandle_importer PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
add_dependencies(cuda_ipc_memhandle_importer tcpgpudmarxd_proto)


# cu ipc memfd handle
set(CU_IPC_MEMFD_HANDLE_SRC
	cu_ipc_memfd_handle.cu
)
add_library(cu_ipc_memfd_handle STATIC ${CU_IPC_MEMFD_HANDLE_SRC})
target_include_directories(
	cu_ipc_memfd_handle
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(cu_ipc_memfd_handle PRIVATE CUDA::cuda_driver)
target_link_libraries(cu_ipc_memfd_handle PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(cu_ipc_memfd_handle PRIVATE cuda_common)
set_target_properties(cu_ipc_memfd_handle PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# cu dmabuf gpu page allocator

set(CU_DMABUF_GPU_ALLOC_SRC
	cu_dmabuf_gpu_page_allocator.cu
)
add_library(cu_dmabuf_gpu_page_allocator STATIC ${CU_DMABUF_GPU_ALLOC_SRC})
target_include_directories(cu_dmabuf_gpu_page_allocator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cu_dmabuf_gpu_page_allocator PRIVATE CUDA::cuda_driver)
target_link_libraries(cu_dmabuf_gpu_page_allocator PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(cu_dmabuf_gpu_page_allocator PRIVATE cuda_common)
set_target_properties(cu_dmabuf_gpu_page_allocator PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# cu ipc memfd handle exporter

set(CU_IPC_MEMFD_HANDLE_EXPORTER_SRC
	cu_ipc_memfd_exporter.cu
)
add_library(cu_ipc_memfd_handle_exporter STATIC ${CU_IPC_MEMFD_HANDLE_EXPORTER_SRC})
target_include_directories(cu_ipc_memfd_handle_exporter PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(cu_ipc_memfd_handle_exporter PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cu_ipc_memfd_handle_exporter PRIVATE CUDA::cuda_driver)
target_link_libraries(cu_ipc_memfd_handle_exporter PRIVATE absl::absl_check absl::log absl::status absl::str_format)
target_link_libraries(cu_ipc_memfd_handle_exporter PUBLIC tcpgpudmarxd_proto)
target_link_libraries(cu_ipc_memfd_handle_exporter PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(cu_ipc_memfd_handle_exporter PRIVATE cuda_common)
target_link_libraries(cu_ipc_memfd_handle_exporter PRIVATE cu_dmabuf_gpu_page_allocator)
target_link_libraries(cu_ipc_memfd_handle_exporter PRIVATE gpu_page_exporter_interface)
target_link_libraries(cu_ipc_memfd_handle_exporter PRIVATE unix_socket_server_lib)
set_target_properties(cu_ipc_memfd_handle_exporter PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# cu ipc memfd handle importer
set(CU_IPC_MEMFD_HANDLE_IMPORTER_SRC
	cu_ipc_memfd_handle_importer.cu
)
add_library(cu_ipc_memfd_handle_importer STATIC ${CU_IPC_MEMFD_HANDLE_IMPORTER_SRC})
target_include_directories(cu_ipc_memfd_handle_importer PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(
	cu_ipc_memfd_handle_importer
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(cu_ipc_memfd_handle_importer PRIVATE CUDA::cuda_driver)
target_link_libraries(cu_ipc_memfd_handle_importer PRIVATE absl::log absl::absl_check absl::status absl::statusor absl::str_format)
target_link_libraries(cu_ipc_memfd_handle_importer PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(cu_ipc_memfd_handle_importer PRIVATE cuda_common)
target_link_libraries(cu_ipc_memfd_handle_importer PRIVATE cu_ipc_memfd_handle)
target_link_libraries(cu_ipc_memfd_handle_importer PRIVATE unix_socket_client_lib)
target_link_libraries(cu_ipc_memfd_handle_importer PRIVATE gpu_mem_importer_telemetry)
set_target_properties(cu_ipc_memfd_handle_importer PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
add_dependencies(cu_ipc_memfd_handle_importer tcpgpudmarxd_proto)

# gpu page exporter factory
add_library(gpu_page_exporter_factory STATIC gpu_page_exporter_factory.cu)
target_include_directories(gpu_page_exporter_factory PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(gpu_page_exporter_factory PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_page_exporter_factory PRIVATE CUDA::cuda_driver)
target_link_libraries(gpu_page_exporter_factory PUBLIC tcpgpudmarxd_proto)
target_link_libraries(gpu_page_exporter_factory PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(gpu_page_exporter_factory PRIVATE cuda_ipc_memhandle_exporter)
target_link_libraries(gpu_page_exporter_factory PRIVATE cu_ipc_memfd_handle_exporter)
set_target_properties(gpu_page_exporter_factory PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

set(client_headers
	${CMAKE_SOURCE_DIR}/cuda/cu_ipc_memfd_handle_importer.cuh
	${CMAKE_SOURCE_DIR}/cuda/cuda_ipc_memhandle_importer.cuh
	${CMAKE_SOURCE_DIR}/cuda/gpu_page_handle_interface.cuh
	${CMAKE_SOURCE_DIR}/cuda/cuda_ipc_memhandle.cuh
)
install(FILES ${client_headers} DESTINATION "${TCPGPUDMARXD_INCLUDE_DIR}/cuda")

set(client_libs
	cuda_common
	cuda_ipc_memhandle
	cu_ipc_memfd_handle
	cuda_ipc_memhandle_importer
	cu_ipc_memfd_handle_importer
)

install(TARGETS ${client_libs} EXPORT gpudirect_tcpxd DESTINATION "${TCPGPUDMARXD_LIB_DIR}")
