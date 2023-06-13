include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
protobuf_generate_cpp(PROTO_SRC PROTO_HEADER
	unix_socket_proto.proto
	unix_socket_message.proto
	gpu_rxq_configuration.proto
)
add_library(proto ${PROTO_HEADER} ${PROTO_SRC})
