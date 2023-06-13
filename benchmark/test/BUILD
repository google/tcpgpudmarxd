load("//third_party/gpus/cuda:build_defs.bzl", "cuda_cc_test", "cuda_library")

cuda_library(
    name = "tcpdirect_single_connection_test_lib",
    srcs = ["tcpdirect_single_connection_test.cu.cc"],
    deps = [
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad:bench_common",
        "//experimental/users/chechenglin/tcpgpudmad:connection_worker",
        "//experimental/users/chechenglin/tcpgpudmad:cuda_common",
        "//experimental/users/chechenglin/tcpgpudmad:cuda_context_manager",
        "//experimental/users/chechenglin/tcpgpudmad:event_handler_factory",
        "//experimental/users/chechenglin/tcpgpudmad:gpu_rxq_configuration_factory",
        "//perftools/gputools/executor/cuda:cuda_platform",
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_cc_test(
    name = "tcpdirect_single_connection_test",
    linkstatic = True,
    deps = [
        ":tcpdirect_single_connection_test_lib",
    ],
)

cuda_library(
    name = "tcpdirect_invalid_ioctl_tests_lib",
    srcs = ["tcpdirect_invalid_ioctl_tests.cu.cc"],
    deps = [
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad:bench_common",
        "//experimental/users/chechenglin/tcpgpudmad:cuda_common",
        "//perftools/gputools/executor/cuda:cuda_platform",
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_cc_test(
    name = "tcpdirect_invalid_ioctl_test",
    linkstatic = True,
    deps = [
        ":tcpdirect_invalid_ioctl_tests_lib",
    ],
)

cuda_library(
    name = "tcpdirect_benchmark_lib",
    srcs = ["tcpdirect_benchmark.cu.cc"],
    deps = [
        "//base:logging",
        "//experimental/users/chechenglin/tcpgpudmad:bench_common",
        "//experimental/users/chechenglin/tcpgpudmad:connection_worker",
        "//experimental/users/chechenglin/tcpgpudmad:cuda_common",
        "//experimental/users/chechenglin/tcpgpudmad:cuda_context_manager",
        "//experimental/users/chechenglin/tcpgpudmad:event_handler_factory",
        "//experimental/users/chechenglin/tcpgpudmad:gpu_rxq_configuration_factory",
        "//perftools/gputools/executor/cuda:cuda_platform",
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_cc_test(
    name = "tcpdirect_benchmark",
    linkstatic = True,
    deps = [
        ":tcpdirect_benchmark_lib",
    ],
)
