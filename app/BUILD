load("//third_party/gpus/cuda:build_defs.bzl", "cuda_cc_test", "cuda_library")
load("//corp/cloud/image_from_binary:build_defs.bzl", "image_from_binary")
load("//third_party/bazel_rules/rules_docker/container:container.bzl", "container_push")

cuda_library(
    name = "tcpgpudmad_lib",
    srcs = ["tcpgpudmad.cu.cc"],
    deps = [
        "//experimental/users/chechenglin/tcpgpudmad:cuda_context_manager",
        "//experimental/users/chechenglin/tcpgpudmad:dmabuf_gpu_page_allocator",
        "//experimental/users/chechenglin/tcpgpudmad:flow_steer_ntuple",
        "//experimental/users/chechenglin/tcpgpudmad:gpu_page_exporter_factory",
        "//experimental/users/chechenglin/tcpgpudmad:gpu_page_exporter_interface",
        "//experimental/users/chechenglin/tcpgpudmad:gpu_rxq_configuration_factory",
        "//experimental/users/chechenglin/tcpgpudmad:nic_configurator_factory",
        "//experimental/users/chechenglin/tcpgpudmad:nic_configurator_interface",
        "//experimental/users/chechenglin/tcpgpudmad:rx_rule_manager",
        "//perftools/gputools/executor/cuda:cuda_platform",
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
        "//third_party/absl/strings:str_format",
        "//third_party/gpus/cuda:cudart_static",
    ],
)

cuda_cc_test(
    name = "tcpgpudmad",
    linkstatic = True,
    deps = [
        ":tcpgpudmad_lib",
    ],
)

image_from_binary(
    name = "tcpgpudmad_image",
    binary = ":tcpgpudmad",
    layers = [
        "//experimental/users/chechenglin/tcpgpudmad:bash_layer",
        "//experimental/users/chechenglin/tcpgpudmad:ethtool_layer",
    ],
)

container_push(
    name = "tcpgpudmad_pusher",
    format = "Docker",
    image = ":tcpgpudmad_image",
    registry = "gcr.io",
    repository = "a3-tcpd-staging-hostpool/chechenglin/tcpgpudmad",
)

container_push(
    name = "tcpgpudmad_pusher_che",
    format = "Docker",
    image = ":tcpgpudmad_image",
    registry = "gcr.io",
    repository = "a3-tcpd-staging-hostpool/che/tcpgpudmad",
)

container_push(
    name = "tcpgpudmad_pusher_mina",
    format = "Docker",
    image = ":tcpgpudmad_image",
    registry = "gcr.io",
    repository = "a3-tcpd-staging-hostpool/mina/tcpgpudmad",
)

container_push(
    name = "tcpgpudmad_pusher_wwchao",
    format = "Docker",
    image = ":tcpgpudmad_image",
    registry = "gcr.io",
    repository = "a3-tcpd-staging-hostpool/wwchao/tcpgpudmad",
)

container_push(
    name = "tcpgpudmad_pusher_kaiyuanz",
    format = "Docker",
    image = ":tcpgpudmad_image",
    registry = "gcr.io",
    repository = "a3-tcpd-staging-hostpool/kaiyuanz/tcpgpudmad",
)

container_push(
    name = "tcpgpudmad_pusher_stable",
    format = "Docker",
    image = ":tcpgpudmad_image",
    registry = "gcr.io",
    repository = "a3-tcpd-staging-hostpool/stable/tcpgpudmad",
)
