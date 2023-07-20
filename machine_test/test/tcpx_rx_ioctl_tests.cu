#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cuda/common.cuh"
#include "include/gpu_rxq_configuration_factory.h"

ABSL_FLAG(std::string, gpu_nic_preset, "auto",
          "The preset configuration for GPU/NIC pairs.  Options: monstertruck, "
          "predvt, auto");
ABSL_FLAG(
    std::string, gpu_nic_topology, "",
    "The path to the textproto file that defines the gpu to nic topology");
ABSL_FLAG(std::string, gpu_nic_topology_proto, "",
          "The string of protobuf that defines the gpu to nic topology");
ABSL_FLAG(std::string, test_name, "all",
          "name of the test to run [invalid_interface, invalid_rxq, "
          "invalid_fd, good, all]");

namespace {
using namespace gpudirect_tcpxd;

struct gpumem_dma_buf_create_info {
  unsigned long gpu_vaddr;
  unsigned long size;
};

struct dma_buf_create_pages_info {
  __u64 pci_bdf[3];
  __s32 dma_buf_fd;
  __s32 create_page_pool;
};

struct dma_buf_frags_bind_rx_queue {
  char ifname[IFNAMSIZ];
  __u32 rxq_idx;
};

#define GPUMEM_DMA_BUF_CREATE _IOW('c', 'c', struct gpumem_dma_buf_create_info)

#define DMA_BUF_BASE 'b'
#define DMA_BUF_CREATE_PAGES \
  _IOW(DMA_BUF_BASE, 2, struct dma_buf_create_pages_info)
#define DMA_BUF_FRAGS_BIND_RX \
  _IOW(DMA_BUF_BASE, 3, struct dma_buf_frags_bind_rx_queue)

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Collect GPU/NIC pair configurations
  std::string gpu_nic_preset = absl::GetFlag(FLAGS_gpu_nic_preset);

  LOG(INFO) << absl::StrFormat(
      "Collecting GPU/NIC pair configurations with preset: %s ...",
      gpu_nic_preset);

  GpuRxqConfigurationList gpu_rxq_configs;
  if (gpu_nic_preset == "manual") {
    if (!absl::GetFlag(FLAGS_gpu_nic_topology_proto).empty() &&
        !absl::GetFlag(FLAGS_gpu_nic_topology).empty()) {
      LOG(FATAL) << "Can't set both gpu_nic_topology_proto and "
                    "gpu_nic_topology at the same time.";
    }
    if (!absl::GetFlag(FLAGS_gpu_nic_topology_proto).empty()) {
      LOG(INFO) << "Getting GPU/NIC topology from text-format proto.";
      gpu_rxq_configs = GpuRxqConfigurationFactory::FromCmdLine(
          absl::GetFlag(FLAGS_gpu_nic_topology_proto));
    } else if (!absl::GetFlag(FLAGS_gpu_nic_topology).empty()) {
      LOG(INFO) << "Getting GPU/NIC toplogy from proto file.";
      gpu_rxq_configs = GpuRxqConfigurationFactory::FromFile(
          absl::GetFlag(FLAGS_gpu_nic_topology));
    } else {
      LOG(FATAL)
          << "Can't select manual option because both gpu_nic_topology_proto "
             "and gpu_nic_topology are empty.";
    }
  } else {
    gpu_rxq_configs = GpuRxqConfigurationFactory::BuildPreset(gpu_nic_preset);
  }
  const char* nic_pci_addr =
      gpu_rxq_configs.gpu_rxq_configs(0).nic_pci_addr().c_str();
  const char* ifname = gpu_rxq_configs.gpu_rxq_configs(0).ifname().c_str();

  int status = 0;

  CU_ASSERT_SUCCESS(cuInit(0));

  CUDA_ASSERT_SUCCESS(cudaSetDevice(0));

  auto test = absl::GetFlag(FLAGS_test_name);

  std::vector<std::string> test_names;

  if (test == "all") {
    test_names.emplace_back("invalid_interface");
    test_names.emplace_back("invalid_rxq");
    test_names.emplace_back("invalid_fd");
    test_names.emplace_back("good");
  } else {
    test_names.emplace_back(test);
  }

  LOG(INFO) << "Allocating GPU memory";

  int dma_buf_fd;
  void* cu_dev_ptr;
  unsigned long alloc_size = 1UL << 21;

  for (const auto& test_name : test_names) {
    CUDA_ASSERT_SUCCESS(cudaMalloc(&cu_dev_ptr, alloc_size));
    cuMemGetHandleForAddressRange((void*)&dma_buf_fd, (CUdeviceptr)cu_dev_ptr,
                                  alloc_size,
                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (dma_buf_fd < 0) {
      PLOG(ERROR) << "cuMemGetHandleForAddressRange() failed!: ";
      CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
      return dma_buf_fd;
    }

    int dma_buf_frags_fd;

    //
    // Bind Queue Tests
    //
    struct dma_buf_create_pages_info frags_create_info;
    frags_create_info.dma_buf_fd = dma_buf_fd;
    frags_create_info.create_page_pool = 1;

    uint16_t pci_bdf[3];
    int ret;
    ret = sscanf(nic_pci_addr, "0000:%hx:%hx.%hx", &pci_bdf[0], &pci_bdf[1],
                 &pci_bdf[2]);
    frags_create_info.pci_bdf[0] = pci_bdf[0];
    frags_create_info.pci_bdf[1] = pci_bdf[1];
    frags_create_info.pci_bdf[2] = pci_bdf[2];

    if (ret != 3) {
      status = -EINVAL;
      close(dma_buf_fd);
      CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
      break;
    }

    dma_buf_frags_fd =
        ioctl(dma_buf_fd, DMA_BUF_CREATE_PAGES, &frags_create_info);
    if (dma_buf_frags_fd < 0) {
      fprintf(stderr, "Error getting dma_buf frags: %s", strerror(errno));
      status = -EIO;
      close(dma_buf_fd);
      CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
      break;
    }

    if (test_name == "invalid_interface") {
      LOG(INFO)
          << "Testing ioctl(DMA_BUF_FRAGS_BIND_RX) with invalid interface ...";
      struct dma_buf_frags_bind_rx_queue bind_cmd;
      bind_cmd.rxq_idx = 1;
      strcpy(bind_cmd.ifname, "invalid");
      int ret = ioctl(dma_buf_frags_fd, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
      if (ret >= 0) {
        LOG(ERROR) << "invalid interfacename should fail!";
        close(ret);
        close(dma_buf_frags_fd);
        close(dma_buf_fd);
        CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
        break;
      }
      LOG(INFO) << "Passed.";
    } else if (test_name == "invalid_rxq") {
      LOG(INFO) << "Testing ioctl(DMA_BUF_FRAGS_BIND_RX) with invalid rxq ...";
      struct dma_buf_frags_bind_rx_queue bind_cmd;
      bind_cmd.rxq_idx = 1000;
      strcpy(bind_cmd.ifname, ifname);
      int ret = ioctl(dma_buf_frags_fd, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
      if (ret >= 0) {
        LOG(ERROR) << "Invalid rxqueue should fail!";
        close(ret);
        close(dma_buf_frags_fd);
        close(dma_buf_fd);
        CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
        break;
      }
      LOG(INFO) << "Passed.";
    } else if (test_name == "invalid_fd") {
      LOG(INFO) << "Testing ioctl(DMA_BUF_FRAGS_BIND_RX) with invalid fd ...";
      struct dma_buf_frags_bind_rx_queue bind_cmd;
      bind_cmd.rxq_idx = 1;
      strcpy(bind_cmd.ifname, ifname);
      int ret = ioctl(dma_buf_frags_fd + 1, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
      if (ret >= 0) {
        LOG(ERROR) << "Invalid dma frag fd should fail!";
        close(ret);
        close(dma_buf_frags_fd);
        close(dma_buf_fd);
        CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
        break;
      }
      LOG(INFO) << "Passed.";
    } else if (test_name == "good") {
      LOG(INFO)
          << "Testing ioctl(DMA_BUF_FRAGS_BIND_RX) with correct parameters ...";
      struct dma_buf_frags_bind_rx_queue bind_cmd;
      bind_cmd.rxq_idx = 1;
      strcpy(bind_cmd.ifname, ifname);
      int ret = ioctl(dma_buf_frags_fd, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
      if (ret < 0) {
        LOG(ERROR) << "Bind queue fail, plz check!";
        close(dma_buf_frags_fd);
        close(dma_buf_fd);
        CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
        break;
      } else {
        close(ret);
      }
      LOG(INFO) << "Passed.";
    } else {
      LOG(ERROR) << "Unknown test: " << test_name;
    }
    close(dma_buf_frags_fd);
    close(dma_buf_fd);
    CUDA_ASSERT_SUCCESS(cudaFree(cu_dev_ptr));
  }
  return status;
}
