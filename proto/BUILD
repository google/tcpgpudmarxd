load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

proto_library(
    name = "unix_socket_message_proto",
    srcs = ["unix_socket_message.proto"],
    deps = ["unix_socket_proto_proto"],
)

cc_proto_library(
    name = "unix_socket_message_cc_proto",
    deps = [":unix_socket_message_proto"],
)

proto_library(
    name = "unix_socket_proto_proto",
    srcs = ["unix_socket_proto.proto"],
)

cc_proto_library(
    name = "unix_socket_proto_cc_proto",
    deps = [":unix_socket_proto_proto"],
)

proto_library(
    name = "gpu_rxq_configuration_proto",
    srcs = ["gpu_rxq_configuration.proto"],
)

cc_proto_library(
    name = "gpu_rxq_configuration_cc_proto",
    deps = [":gpu_rxq_configuration_proto"],
)
