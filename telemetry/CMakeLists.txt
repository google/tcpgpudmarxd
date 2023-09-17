ADD_SUBDIRECTORY(proto)

# telemetry base
add_library(telemetry_interface STATIC telemetry_interface.cc)
target_include_directories(telemetry_interface PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(telemetry_interface PUBLIC grpc++)
target_link_libraries(telemetry_interface PRIVATE tcpgpudmarxd_telemetry_proto)
target_link_libraries(telemetry_interface PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(telemetry_interface PUBLIC absl::absl_check absl::flags absl::flags_parse absl::log absl::str_format absl::debugging absl::debugging_internal absl::failure_signal_handler)
add_dependencies(telemetry_interface tcpgpudmarxd_telemetry_proto)

# rx rule client telemetry
add_library(rx_rule_client_telemetry STATIC rx_rule_client_telemetry.cc)
target_include_directories(rx_rule_client_telemetry PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_client_telemetry PUBLIC grpc++)
target_link_libraries(rx_rule_client_telemetry PRIVATE tcpgpudmarxd_telemetry_proto)
target_link_libraries(rx_rule_client_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_client_telemetry PUBLIC telemetry_interface)
add_dependencies(rx_rule_client_telemetry tcpgpudmarxd_telemetry_proto)



# rx rule manager telemetry
add_library(rx_rule_manager_telemetry STATIC rx_rule_manager_telemetry.cc)
target_include_directories(rx_rule_manager_telemetry PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_manager_telemetry PUBLIC grpc++)
target_link_libraries(rx_rule_manager_telemetry PUBLIC tcpgpudmarxd_telemetry_proto)
target_link_libraries(rx_rule_manager_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_manager_telemetry PUBLIC telemetry_interface)


# gpu mem importer telemetry
add_library(gpu_mem_importer_telemetry STATIC gpu_mem_importer_telemetry.cc)
target_include_directories(gpu_mem_importer_telemetry PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(gpu_mem_importer_telemetry PUBLIC grpc++)
target_link_libraries(gpu_mem_importer_telemetry PUBLIC tcpgpudmarxd_telemetry_proto)
target_link_libraries(gpu_mem_importer_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(gpu_mem_importer_telemetry PUBLIC telemetry_interface)
add_dependencies(gpu_mem_importer_telemetry tcpgpudmarxd_telemetry_proto)

# gpu mem exporter telemetry
add_library(gpu_mem_exporter_telemetry STATIC gpu_mem_exporter_telemetry.cc)
target_include_directories(gpu_mem_exporter_telemetry PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC grpc++)
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC tcpgpudmarxd_telemetry_proto)
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC telemetry_interface)