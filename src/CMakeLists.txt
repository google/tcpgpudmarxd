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
	pci_helpers.cc
)

add_library(gpu_rxq_configurator STATIC ${GPU_RXQ_CONFIG_SRC})
target_include_directories(gpu_rxq_configurator PRIVATE cuda)
target_include_directories(gpu_rxq_configurator PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(gpu_rxq_configurator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_rxq_configurator PRIVATE CUDA::cuda_driver)
target_link_libraries(gpu_rxq_configurator PRIVATE absl::log absl::flags absl::status absl::str_format)
target_link_libraries(gpu_rxq_configurator PUBLIC tcpgpudmarxd_proto)
target_link_libraries(gpu_rxq_configurator PUBLIC ${Protobuf_LIBRARIES})
set_target_properties(gpu_rxq_configurator PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# NIC configurator

set(NIC_CONFIGURATOR_SRC
	ioctl_nic_configurator.cc
	dummy_ethtool_nic_configurator.cc
	ethtool_no_headersplit_nic_configurator.cc
	ethtool_nic_configurator.cc
	nic_configurator_factory.cc
)

add_library(nic_configurator STATIC ${NIC_CONFIGURATOR_SRC})
target_include_directories(nic_configurator PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(nic_configurator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(nic_configurator PRIVATE absl::absl_check absl::status absl::str_format)
target_link_libraries(nic_configurator PUBLIC tcpgpudmarxd_proto)
target_link_libraries(nic_configurator PUBLIC ${Protobuf_LIBRARIES})

# rxq dma page exporter interface

add_library(gpu_page_exporter_interface STATIC
	gpu_page_exporter_interface.cc)
target_include_directories(gpu_page_exporter_interface PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(gpu_page_exporter_interface PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(gpu_page_exporter_interface PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(gpu_page_exporter_interface PUBLIC tcpgpudmarxd_proto)
target_link_libraries(gpu_page_exporter_interface PUBLIC ${Protobuf_LIBRARIES})

# unix socket lib

set(UNIX_SOCKET_LIB_SRC
	unix_socket_client.cc
	unix_socket_connection.cc
	unix_socket_server.cc
)

add_library(unix_socket_lib STATIC ${UNIX_SOCKET_LIB_SRC})
target_include_directories(unix_socket_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(unix_socket_lib PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(unix_socket_lib PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(unix_socket_lib PUBLIC tcpgpudmarxd_proto)
target_link_libraries(unix_socket_lib PUBLIC ${Protobuf_LIBRARIES})

# rx rule client
add_library(rx_rule_client STATIC rx_rule_client.cc)
target_include_directories(rx_rule_client PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(rx_rule_client PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_client PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(rx_rule_client PUBLIC tcpgpudmarxd_proto)
target_link_libraries(rx_rule_client PUBLIC ${Protobuf_LIBRARIES})


# rx rule manager
add_library(rx_rule_manager STATIC rx_rule_manager.cc)
target_include_directories(rx_rule_manager PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(rx_rule_manager PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_manager PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(rx_rule_manager PUBLIC tcpgpudmarxd_proto)
target_link_libraries(rx_rule_manager PUBLIC ${Protobuf_LIBRARIES})
