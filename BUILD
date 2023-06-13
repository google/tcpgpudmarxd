load("//third_party/gpus/cuda:build_defs.bzl", "cuda_library")
load("//corp/cloud/image_from_binary:build_defs.bzl", "image_from_binary")
load("//third_party/bazel_rules/rules_docker/container:container.bzl", "container_layer", "container_push")

cc_library(
    name = "nic_configurator_interface",
    hdrs = ["include/nic_configurator_interface.h"],
    deps = [
        ":flow_steer_ntuple",
        "//third_party/absl/status",
    ],
)

cc_library(
    name = "ioctl_nic_configurator",
    srcs = ["src/ioctl_nic_configurator.cc"],
    hdrs = ["include/ioctl_nic_configurator.h"],
    deps = [
        ":flow_steer_ntuple",
        ":include/ethtool_common",
        ":nic_configurator_interface",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "ethtool_nic_configurator",
    srcs = ["src/ethtool_nic_configurator.cc"],
    hdrs = ["include/ethtool_nic_configurator.h"],
    deps = [
        ":flow_steer_ntuple",
        ":nic_configurator_interface",
        "//base:logging",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "dummy_ethtool_nic_configurator",
    srcs = ["src/dummy_ethtool_nic_configurator.cc"],
    hdrs = ["include/dummy_ethtool_nic_configurator.h"],
    deps = [
        ":ethtool_nic_configurator",
        "//base:logging",
        "//third_party/absl/status",
    ],
)

cc_library(
    name = "ethtool_no_headersplit_nic_configurator",
    srcs = ["src/ethtool_no_headersplit_nic_configurator.cc"],
    hdrs = ["include/ethtool_no_headersplit_nic_configurator.h"],
    deps = [
        ":ethtool_nic_configurator",
        ":flow_steer_ntuple",
        ":nic_configurator_interface",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "nic_configurator_factory",
    srcs = ["src/nic_configurator_factory.cc"],
    hdrs = ["include/nic_configurator_factory.h"],
    deps = [
        ":dummy_ethtool_nic_configurator",
        ":ethtool_nic_configurator",
        ":ethtool_no_headersplit_nic_configurator",
        ":ioctl_nic_configurator",
        ":nic_configurator_interface",
    ],
)

cc_library(
    name = "gpu_rxq_configurator_interface",
    hdrs = ["include/gpu_rxq_configurator_interface.h"],
    deps = ["//experimental/users/chechenglin/tcpgpudmad/proto:gpu_rxq_configuration_cc_proto"],
)

cc_library(
    name = "monstertruck_gpu_rxq_configurator",
    srcs = ["src/monstertruck_gpu_rxq_configurator.cc"],
    hdrs = ["include/monstertruck_gpu_rxq_configurator.h"],
    deps = [
        ":gpu_rxq_configurator_interface",
    ],
)

cc_library(
    name = "auto_discovery_gpu_rxq_configurator",
    srcs = ["src/auto_discovery_gpu_rxq_configurator.cc"],
    hdrs = ["include/auto_discovery_gpu_rxq_configurator.h"],
    deps = [
        ":a3_gpu_rxq_configurator",
        ":gpu_rxq_configurator_interface",
    ],
)

cc_library(
    name = "predvt_gpu_rxq_configurator",
    srcs = ["src/predvt_gpu_rxq_configurator.cc"],
    hdrs = ["include/predvt_gpu_rxq_configurator.h"],
    deps = [
        ":gpu_rxq_configurator_interface",
    ],
)

cc_library(
    name = "a3vm_gpu_rxq_configurator",
    srcs = ["src/a3vm_gpu_rxq_configurator.cc"],
    hdrs = ["include/a3vm_gpu_rxq_configurator.h"],
    deps = [
        ":gpu_rxq_configurator_interface",
    ],
)

cc_library(
    name = "gpu_rxq_configuration_factory",
    srcs = ["src/gpu_rxq_configuration_factory.cc"],
    hdrs = ["include/gpu_rxq_configuration_factory.h"],
    deps = [
        ":a3vm_gpu_rxq_configurator",
        ":auto_discovery_gpu_rxq_configurator",
        ":gpu_rxq_configurator_interface",
        ":monstertruck_gpu_rxq_configurator",
        ":predvt_gpu_rxq_configurator",
        "//experimental/users/chechenglin/tcpgpudmad/proto:gpu_rxq_configuration_cc_proto",
        "//third_party/absl/log",
        "//third_party/protobuf",
    ],
)

cc_library(
    name = "pci_helpers",
    srcs = ["src/pci_helpers.cc"],
    hdrs = ["include/pci_helpers.h"],
    deps = [
        "//third_party/absl/log",
    ],
)

cuda_library(
    name = "a3_gpu_rxq_configurator",
    srcs = ["src/a3_gpu_rxq_configurator.cu.cc"],
    hdrs = ["include/a3_gpu_rxq_configurator.cu.h"],
    deps = [
        ":cuda_common",
        ":gpu_rxq_configurator_interface",
        ":pci_helpers",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/flags:flag",
        "//third_party/absl/log",
        "//third_party/absl/strings",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cc_library(
    name = "unix_socket_connection",
    srcs = ["src/unix_socket_connection.cc"],
    hdrs = ["include/unix_socket_connection.h"],
    deps = [
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_proto_cc_proto",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "unix_socket_server",
    srcs = ["src/unix_socket_server.cc"],
    hdrs = ["include/unix_socket_server.h"],
    deps = [
        ":unix_socket_connection",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "unix_socket_client",
    srcs = ["src/unix_socket_client.cc"],
    hdrs = ["include/unix_socket_client.h"],
    deps = [
        ":unix_socket_connection",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:str_format",
    ],
)

cc_test(
    name = "unix_socket_server_client_test",
    srcs = ["test/unix_socket_server_client_test.cc"],
    deps = [
        ":unix_socket_client",
        ":unix_socket_server",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "flow_steer_ntuple",
    hdrs = ["include/flow_steer_ntuple.h"],
)

cc_library(
    name = "rx_rule_manager",
    srcs = ["src/rx_rule_manager.cc"],
    hdrs = ["include/rx_rule_manager.h"],
    deps = [
        ":flow_steer_ntuple",
        ":nic_configurator_interface",
        ":unix_socket_server",
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad/proto:gpu_rxq_configuration_cc_proto",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/functional:bind_front",
        "//third_party/absl/hash",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "rx_rule_client",
    srcs = ["src/rx_rule_client.cc"],
    hdrs = ["include/rx_rule_client.h"],
    deps = [
        ":flow_steer_ntuple",
        ":unix_socket_client",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/status",
    ],
)

cuda_library(
    name = "cuda_common",
    srcs = ["cuda/common.cu.cc"],
    hdrs = ["cuda/common.cu.h"],
    deps = [
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "cuda_context_manager",
    srcs = ["cuda/cuda_context_manager.cu.cc"],
    hdrs = ["cuda/cuda_context_manager.cu.h"],
    deps = [
        ":cuda_common",
        "//base:logging",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "dmabuf_gpu_page_allocator",
    srcs = ["cuda/dmabuf_gpu_page_allocator.cu.cc"],
    hdrs = [
        "cuda/dmabuf_gpu_page_allocator.cu.h",
        "cuda/gpu_page_allocator_interface.cu.h",
    ],
    deps = [
        ":cuda_common",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "cu_dmabuf_gpu_page_allocator",
    srcs = ["cuda/cu_dmabuf_gpu_page_allocator.cu.cc"],
    hdrs = [
        "cuda/cu_dmabuf_gpu_page_allocator.cu.h",
        "cuda/gpu_page_allocator_interface.cu.h",
        "include/ipc_gpumem_fd_metadata.h",
    ],
    deps = [
        ":cuda_common",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cc_library(
    name = "gpu_page_exporter_interface",
    srcs = ["src/gpu_page_exporter_interface.cc"],
    hdrs = ["include/gpu_page_exporter_interface.h"],
    deps = [
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad/proto:gpu_rxq_configuration_cc_proto",
        "//third_party/absl/status",
    ],
)

cuda_library(
    name = "cu_ipc_memfd_exporter",
    srcs = ["cuda/cu_ipc_memfd_exporter.cu.cc"],
    hdrs = [
        "cuda/cu_ipc_memfd_exporter.cu.h",
        "include/ipc_gpumem_fd_metadata.h",
    ],
    deps = [
        ":cu_dmabuf_gpu_page_allocator",
        ":cuda_common",
        ":cuda_context_manager",
        ":gpu_page_exporter_interface",
        ":unix_socket_server",
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad/proto:gpu_rxq_configuration_cc_proto",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
    ],
)

cuda_library(
    name = "cuda_ipc_memhandle_exporter",
    srcs = ["cuda/cuda_ipc_memhandle_exporter.cu.cc"],
    hdrs = [
        "cuda/cuda_ipc_memhandle_exporter.cu.h",
    ],
    deps = [
        ":cuda_common",
        ":cuda_context_manager",
        ":dmabuf_gpu_page_allocator",
        ":gpu_page_exporter_interface",
        ":unix_socket_server",
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad/proto:gpu_rxq_configuration_cc_proto",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
    ],
)

cuda_library(
    name = "gpu_page_handle_interface",
    hdrs = ["cuda/gpu_page_handle_interface.cu.h"],
)

cuda_library(
    name = "cu_ipc_memfd_handle",
    srcs = ["cuda/cu_ipc_memfd_handle.cu.cc"],
    hdrs = ["cuda/cu_ipc_memfd_handle.cu.h"],
    deps = [
        ":cuda_common",
        ":gpu_page_handle_interface",
        "//base:logging",
        "//third_party/absl/strings:str_format",
    ],
)

cuda_library(
    name = "cu_ipc_memfd_handle_importer",
    srcs = ["cuda/cu_ipc_memfd_handle_importer.cu.cc"],
    hdrs = [
        "cuda/cu_ipc_memfd_handle_importer.cu.h",
        "include/ipc_gpumem_fd_metadata.h",
    ],
    deps = [
        ":cu_ipc_memfd_handle",
        ":cuda_common",
        ":gpu_page_handle_interface",
        ":unix_socket_client",
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
    ],
)

cuda_library(
    name = "cuda_ipc_memhandle",
    srcs = ["cuda/cuda_ipc_memhandle.cu.cc"],
    hdrs = ["cuda/cuda_ipc_memhandle.cu.h"],
    deps = [
        ":cuda_common",
        ":gpu_page_handle_interface",
        "//base:logging",
    ],
)

cuda_library(
    name = "cuda_ipc_memhandle_importer",
    srcs = ["cuda/cuda_ipc_memhandle_importer.cu.cc"],
    hdrs = [
        "cuda/cuda_ipc_memhandle_importer.cu.h",
    ],
    deps = [
        ":cuda_ipc_memhandle",
        ":gpu_page_handle_interface",
        ":unix_socket_client",
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
    ],
)

cuda_library(
    name = "gpu_page_exporter_factory",
    srcs = ["cuda/gpu_page_exporter_factory.cu.cc"],
    hdrs = ["cuda/gpu_page_exporter_factory.cu.h"],
    deps = [
        ":cu_ipc_memfd_exporter",
        ":cuda_ipc_memhandle_exporter",
        ":gpu_page_exporter_interface",
    ],
)

cc_library(
    name = "bench_common",
    srcs = [
        "benchmark/benchmark_common.cc",
        "benchmark/socket_helper.cc",
    ],
    hdrs = [
        "benchmark/benchmark_common.h",
        "benchmark/socket_helper.h",
    ],
    deps = [
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
    ],
)

cuda_library(
    name = "validation",
    srcs = ["benchmark/validation.cu.cc"],
    hdrs = ["benchmark/validation.cu.h"],
    deps = [
        ":cuda_common",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
    ],
)

cuda_library(
    name = "gpu_send_event_handler",
    srcs = ["benchmark/gpu_send_event_handler.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_send_event_handler.cu.h",
        "benchmark/tcpdirect_common.h",
        "cuda/gpu_page_allocator_interface.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":validation",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_receive_common",
    srcs = ["benchmark/gpu_receive_common.cu.cc"],
    hdrs = [
        "benchmark/gpu_receive_common.cu.h",
        "benchmark/tcpdirect_common.h",
    ],
    deps = [
        ":cuda_common",
        ":cuda_ipc_memhandle_importer",
        ":gpu_page_handle_interface",
        ":unix_socket_client",
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad/proto:unix_socket_message_cc_proto",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_receive_event_handler",
    srcs = ["benchmark/gpu_receive_event_handler.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_receive_event_handler.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":gpu_receive_common",
        ":validation",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_receive_miss_flag_event_handler",
    srcs = ["benchmark/gpu_receive_miss_flag_event_handler.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_receive_miss_flag_event_handler.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":gpu_receive_common",
        ":validation",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_receive_mix_tcp_event_handler",
    srcs = ["benchmark/gpu_receive_mix_tcp_event_handler.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_receive_mix_tcp_event_handler.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":gpu_receive_common",
        ":validation",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_receive_token_free_event_handler",
    srcs = ["benchmark/gpu_receive_token_free_event_handler.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_receive_token_free_event_handler.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":gpu_receive_common",
        ":validation",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_receive_no_token_free_event_handler",
    srcs = ["benchmark/gpu_receive_no_token_free_event_handler.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_receive_no_token_free_event_handler.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":gpu_receive_common",
        ":validation",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_send_event_handler_miss_flag",
    srcs = ["benchmark/gpu_send_event_handler_miss_flag.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_send_event_handler_miss_flag.cu.h",
        "benchmark/tcpdirect_common.h",
        "cuda/gpu_page_allocator_interface.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":validation",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_send_event_handler_mix_tcp",
    srcs = ["benchmark/gpu_send_event_handler_mix_tcp.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_send_event_handler_mix_tcp.cu.h",
        "benchmark/tcpdirect_common.h",
        "cuda/gpu_page_allocator_interface.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":validation",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_library(
    name = "gpu_send_oob_event_handler",
    srcs = ["benchmark/gpu_send_oob_event_handler.cu.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/gpu_send_oob_event_handler.cu.h",
        "benchmark/tcpdirect_common.h",
        "cuda/gpu_page_allocator_interface.cu.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_common",
        ":validation",
        "//base",
        "//base:logging",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cc_library(
    name = "tcp_receive_event_handler",
    srcs = ["benchmark/tcp_receive_event_handler.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/tcp_receive_event_handler.h",
    ],
    deps = [
        ":validation",
        "//base:logging",
    ],
)

cc_library(
    name = "tcp_receive_tcp_direct_event_handler",
    srcs = ["benchmark/tcp_receive_tcp_direct_event_handler.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/tcp_receive_tcp_direct_event_handler.h",
    ],
    deps = [
        ":validation",
        "//base:logging",
    ],
)

cc_library(
    name = "tcp_send_event_handler",
    srcs = ["benchmark/tcp_send_event_handler.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/tcp_send_event_handler.h",
    ],
    deps = [
        ":validation",
        "//base:logging",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "tcp_send_tcp_direct_event_handler",
    srcs = ["benchmark/tcp_send_tcp_direct_event_handler.cc"],
    hdrs = [
        "benchmark/event_handler_interface.h",
        "benchmark/tcp_send_tcp_direct_event_handler.h",
    ],
    deps = [
        ":validation",
        "//base:logging",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "include/ethtool_common",
    hdrs = ["include/ethtool_common.h"],
)

cuda_library(
    name = "event_handler_factory",
    srcs = ["benchmark/event_handler_factory.cu.cc"],
    hdrs = [
        "benchmark/event_handler_factory.cu.h",
        "benchmark/event_handler_interface.h",
    ],
    deps = [
        ":dmabuf_gpu_page_allocator",
        ":gpu_receive_event_handler",
        ":gpu_receive_miss_flag_event_handler",
        ":gpu_receive_mix_tcp_event_handler",
        ":gpu_receive_no_token_free_event_handler",
        ":gpu_receive_token_free_event_handler",
        ":gpu_send_event_handler",
        ":gpu_send_event_handler_miss_flag",
        ":gpu_send_event_handler_mix_tcp",
        ":gpu_send_oob_event_handler",
        ":tcp_receive_event_handler",
        ":tcp_receive_tcp_direct_event_handler",
        ":tcp_send_event_handler",
        ":tcp_send_tcp_direct_event_handler",
        "//base:logging",
    ],
)

cuda_library(
    name = "connection_worker",
    srcs = ["benchmark/connection_worker.cu.cc"],
    hdrs = [
        "benchmark/connection_worker.cu.h",
        "benchmark/event_handler_interface.h",
    ],
    deps = [
        ":bench_common",
        ":cuda_context_manager",
        ":event_handler_factory",
        "//base:logging",
        "//third_party/absl/strings:str_format",
    ],
)

cc_binary(
    name = "ethtool_test",
    srcs = ["ethtool_test/ethtool_test.cc"],
    deps = [
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
        "//third_party/absl/strings",
    ],
)

sh_binary(
    name = "run_ethtool",
    srcs = ["ethtool_test/run_ethtool.sh"],
)

genrule(
    name = "install_ethtool",
    srcs = ["//third_party/prodimage/modules/ethtool:ethtool_msv_mpm_pkg_tarball.tar"],
    outs = ["ethtool"],
    cmd = "tar -xf $<; cp sbin/ethtool $@",
)

container_layer(
    name = "ethtool_layer",
    files = [
        ":install_ethtool",
    ],
    symlinks = {
        "/sbin/ethtool": "../ethtool",
    },
)

container_layer(
    name = "bash_layer",
    env = {
        "PATH": "/bin:/sbin",
        "SHELL": "/bin/bash",
    },
    files = [
        "//third_party/bash:bash5",
    ],
    symlinks = {
        "/bin/bash": "../bash5",
        "/bin/sh": "../bash5",
    },
)

image_from_binary(
    name = "test_image_ethtool",
    binary = ":run_ethtool",
    layers = [
        ":bash_layer",
        ":ethtool_layer",
    ],
)

container_push(
    name = "test_image_ethtool_pusher",
    format = "Docker",
    image = ":test_image_ethtool",
    repository = "wwchao/ethtool_image",
)
