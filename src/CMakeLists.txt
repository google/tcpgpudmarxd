# gpu rxq configuration

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

set(GPU_RXQ_CONFIG_SRC
	a3_gpu_rxq_configurator.cu
	a3vm_gpu_rxq_configurator.cc
	auto_discovery_gpu_rxq_configurator.cc
	gpu_rxq_configuration_factory.cc
	monstertruck_gpu_rxq_configurator.cc
	predvt_gpu_rxq_configurator.cc
)

set(GPU_RXQ_CONFIG_HEADER
	${CMAKE_SOURCE_DIR}/include/a3_gpu_rxq_configurator.cuh
	${CMAKE_SOURCE_DIR}/include/a3vm_gpu_rxq_configurator.h
	${CMAKE_SOURCE_DIR}/include/auto_discovery_gpu_rxq_configurator.h
	${CMAKE_SOURCE_DIR}/include/gpu_rxq_configuration_factory.h
	${CMAKE_SOURCE_DIR}/include/gpu_rxq_configurator_interface.h
	${CMAKE_SOURCE_DIR}/include/monstertruck_gpu_rxq_configurator.h
	${CMAKE_SOURCE_DIR}/include/predvt_gpu_rxq_configurator.h
)

add_library(gpu_rxq_configurator STATIC ${GPU_RXQ_CONFIG_SRC} ${GPU_RXQ_CONFIG_HEADER})
target_include_directories(gpu_rxq_configurator PRIVATE cuda)
target_include_directories(gpu_rxq_configurator PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(gpu_rxq_configurator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_rxq_configurator PRIVATE CUDA::cuda_driver)
target_link_libraries(gpu_rxq_configurator PRIVATE absl::base absl::flags absl::status absl::str_format)
target_link_libraries(gpu_rxq_configurator PUBLIC tcpgpudmarxd_proto)
set_target_properties(gpu_rxq_configurator PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
