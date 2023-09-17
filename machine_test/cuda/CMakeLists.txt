enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

link_libraries(tcpgpudmarxd_telemetry_proto)

add_library(connection_worker STATIC connection_worker.cu)
target_include_directories(connection_worker PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(connection_worker PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(connection_worker PRIVATE CUDA::cuda_driver)
target_link_libraries(connection_worker PRIVATE absl::log absl::absl_check absl::status absl::str_format)
target_link_libraries(connection_worker PRIVATE cuda_common)
target_link_libraries(connection_worker PUBLIC rx_rule_client)
target_link_libraries(connection_worker PUBLIC tcpgpudmarxd_proto)
target_link_libraries(connection_worker PUBLIC ${Protobuf_LIBRARIES})
set_target_properties(connection_worker PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_library(gpu_receive_common STATIC gpu_receive_common.cu)
target_include_directories(gpu_receive_common PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(gpu_receive_common PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_receive_common PRIVATE CUDA::cuda_driver)
target_link_libraries(gpu_receive_common PRIVATE absl::log absl::status)
target_link_libraries(gpu_receive_common PRIVATE cuda_common)
target_link_libraries(gpu_receive_common PRIVATE cuda_ipc_memhandle_importer)
target_link_libraries(gpu_receive_common PUBLIC tcpgpudmarxd_proto)
target_link_libraries(gpu_receive_common PUBLIC ${Protobuf_LIBRARIES})
set_target_properties(gpu_receive_common PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_library(validation STATIC validation.cu)
target_include_directories(validation PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(validation PRIVATE CUDA::cuda_driver)
target_link_libraries(validation PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(validation PRIVATE cuda_common)
set_target_properties(validation PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_library(cuda_dmabuf_gpu_page_allocator STATIC cuda_dmabuf_gpu_page_allocator.cu)
target_include_directories(cuda_dmabuf_gpu_page_allocator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cuda_dmabuf_gpu_page_allocator PRIVATE CUDA::cuda_driver)
target_link_libraries(cuda_dmabuf_gpu_page_allocator PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(cuda_dmabuf_gpu_page_allocator PRIVATE cuda_common)
set_target_properties(cuda_dmabuf_gpu_page_allocator PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

set(GPU_TCP_EVENT_HANDLERS_SRC
    gpu_receive_event_handler.cu
    gpu_receive_miss_flag_event_handler.cu
    gpu_receive_mix_tcp_event_handler.cu
    gpu_receive_no_token_free_event_handler.cu
    gpu_receive_token_free_event_handler.cu
    gpu_send_event_handler.cu
    gpu_send_event_handler_miss_flag.cu
    gpu_send_event_handler_mix_tcp.cu
    gpu_send_oob_event_handler.cu
)

add_library(gpu_tcp_event_handlers STATIC ${GPU_TCP_EVENT_HANDLERS_SRC})
target_include_directories(gpu_tcp_event_handlers PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR})
target_link_libraries(gpu_tcp_event_handlers PRIVATE CUDA::cuda_driver)
target_link_libraries(gpu_tcp_event_handlers PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(gpu_tcp_event_handlers PRIVATE cuda_common)
target_link_libraries(gpu_tcp_event_handlers PRIVATE cuda_dmabuf_gpu_page_allocator)
target_link_libraries(gpu_tcp_event_handlers PRIVATE gpu_receive_common)
target_link_libraries(gpu_tcp_event_handlers PRIVATE dmabuf_gpu_page_allocator)
target_link_libraries(gpu_tcp_event_handlers PRIVATE cu_ipc_memfd_handle_importer)
set_target_properties(gpu_tcp_event_handlers PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_library(event_handler_factory STATIC event_handler_factory.cu)
target_include_directories(event_handler_factory PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR})
target_link_libraries(event_handler_factory PRIVATE CUDA::cuda_driver)
target_link_libraries(event_handler_factory PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(event_handler_factory PRIVATE cuda_common)
target_link_libraries(event_handler_factory PRIVATE gpu_tcp_event_handlers)
target_link_libraries(event_handler_factory PRIVATE tcp_event_handlers)
set_target_properties(event_handler_factory PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
