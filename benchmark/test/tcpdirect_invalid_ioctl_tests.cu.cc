#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstdint>
#include <string>
#include <unordered_map>

#include "base/logging.h"
#include "experimental/users/chechenglin/tcpgpudmad/cuda/common.cu.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/flags/parse.h"
#include "third_party/absl/strings/str_format.h"

ABSL_FLAG(std::string, test_name, "", "name of the test to run");

namespace {

inline constexpr const char* gpu_pci_addr = "0000:6a:00.0";
inline constexpr const char* nic_pci_addr = "0000:6b:00.0";
inline constexpr const char* ifname = "hpn1";
inline constexpr const char* rx_queues_str = "1";

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

const char kNvp2pDmabufProcfsPrefix[] = "/proc/driver/nvp2p_dma_buf";
}  // namespace
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  int status = 0;
  int num_gpu = 0;
  int fd;

  CU_ASSERT_SUCCESS(cuInit(0));
  CUDA_ASSERT_SUCCESS(cudaGetDeviceCount(&num_gpu));

  assert(num_gpu > 0);

  std::unordered_map<std::string, int> pci_addr_to_gpu_idx;

  for (int i = 0; i < num_gpu; i++) {
    cudaDeviceProp prop;
    CUDA_ASSERT_SUCCESS(cudaGetDeviceProperties(&prop, i));
    std::string gpu_pci_addr = absl::StrFormat(
        "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    pci_addr_to_gpu_idx[gpu_pci_addr] = i;
  }

  CU_ASSERT_SUCCESS(cuDeviceGet(&fd, pci_addr_to_gpu_idx[gpu_pci_addr]));

  CUcontext ctx;
  CU_ASSERT_SUCCESS(cuCtxCreate(&ctx, 0, fd));
  CU_ASSERT_SUCCESS(cuCtxPushCurrent(ctx));

  LOG(INFO) << "Allocating GPU memory";
  unsigned long alloc_size = 1UL << 21;
  CUdeviceptr p;
  CU_ASSERT_SUCCESS(cuMemAlloc(&p, alloc_size));
  CU_ASSERT_SUCCESS(cuMemsetD32(p, 0, alloc_size / 4));

  LOG(INFO) << "Done allocating GPU memory";

  auto test_name = absl::GetFlag(FLAGS_test_name);
  int dma_buf_fd, dma_buf_frags_fd;

  std::string procfs_fd_path =
      absl::StrFormat("%s/%s/new_fd", kNvp2pDmabufProcfsPrefix, gpu_pci_addr);

  int procfs_fd = open(procfs_fd_path.c_str(), O_WRONLY);
  if (procfs_fd == -1) {
    LOG(ERROR) << absl::StrFormat("Error opening %s", procfs_fd_path);
    goto out;
  }

  //
  // GPUMEM_DMA_BUF_CREATE tests
  //
  if (test_name == "invalid_gpumem_create") {
    struct gpumem_dma_buf_create_info create_info = {p - 32, alloc_size};
    int ret = ioctl(procfs_fd, GPUMEM_DMA_BUF_CREATE, &create_info);
    if (!ret) {
      LOG(ERROR) << "Invalid pointer should fail.";
      close(ret);
    }
    goto close_procfs_fd;
  } else if (test_name == "incorrect_gpumem_size") {
    struct gpumem_dma_buf_create_info create_info = {p, alloc_size << 1};
    int ret = ioctl(procfs_fd, GPUMEM_DMA_BUF_CREATE, &create_info);
    if (!ret) {
      LOG(ERROR) << "Dishonest size should fail.";
      close(ret);
    }
    goto close_procfs_fd;
  }

  struct gpumem_dma_buf_create_info create_info;
  create_info = {p, alloc_size};
  dma_buf_fd = ioctl(procfs_fd, GPUMEM_DMA_BUF_CREATE, &create_info);

  if (dma_buf_fd < 0) {
    PLOG(ERROR) << "ioctl(GPUMEM_DMA_BUF_CREATE): error.";
    goto close_procfs_fd;
  }

  //
  // DMA_BUF_CREATE_PAGES tests
  //
  if (test_name == "invalid_pci_frag_create") {
    struct dma_buf_create_pages_info frags_create_info;
    frags_create_info.dma_buf_fd = dma_buf_fd;
    frags_create_info.create_page_pool = 0;

    frags_create_info.pci_bdf[0] = -1;
    frags_create_info.pci_bdf[1] = -2;
    frags_create_info.pci_bdf[2] = -3;

    for (int i = 0; i < 2; ++i) {
      frags_create_info.create_page_pool = i;
      int ret = ioctl(dma_buf_fd, DMA_BUF_CREATE_PAGES, &frags_create_info);
      if (ret >= 0) {
        LOG(ERROR) << "Invalid pci address for the NIC should fail. (TX)";
        close(ret);
        goto close_dmabuf_fd;
      }
    }
    goto close_dmabuf_fd;
  } else if (test_name == "invalid_dmabuf_fd_frag_create") {
    int wrong_dma_buf_fd = dma_buf_fd * 2;
    struct dma_buf_create_pages_info frags_create_info;
    frags_create_info.dma_buf_fd = wrong_dma_buf_fd;
    frags_create_info.create_page_pool = 0;
    uint16_t pci_bdf[3];
    int ret = sscanf(nic_pci_addr, "0000:%hx:%hx.%hx", &pci_bdf[0], &pci_bdf[1],
                     &pci_bdf[2]);
    frags_create_info.pci_bdf[0] = pci_bdf[0];
    frags_create_info.pci_bdf[1] = pci_bdf[1];
    frags_create_info.pci_bdf[2] = pci_bdf[2];

    for (int i = 0; i < 2; ++i) {
      frags_create_info.create_page_pool = i;
      ret = ioctl(dma_buf_fd, DMA_BUF_CREATE_PAGES, &frags_create_info);
      if (ret >= 0) {
        LOG(ERROR) << "Invalid dma_buf_fd should fail.";
        close(ret);
        goto close_dmabuf_fd;
      }

      ret = ioctl(wrong_dma_buf_fd, DMA_BUF_CREATE_PAGES, &frags_create_info);
      if (ret >= 0) {
        LOG(ERROR) << "Invalid dma_buf_fd should fail.";
        close(ret);
        goto close_dmabuf_fd;
      }
    }
    goto close_dmabuf_fd;
  }

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
    goto close_dmabuf_fd;
  }

  dma_buf_frags_fd =
      ioctl(dma_buf_fd, DMA_BUF_CREATE_PAGES, &frags_create_info);
  if (dma_buf_frags_fd < 0) {
    fprintf(stderr, "Error getting dma_buf frags: %s", strerror(errno));
    status = -EIO;
    goto close_dmabuf_fd;
  }

  if (test_name == "invalid_interface") {
    struct dma_buf_frags_bind_rx_queue bind_cmd;
    bind_cmd.rxq_idx = 1;
    strcpy(bind_cmd.ifname, "invalid");
    int ret = ioctl(dma_buf_frags_fd, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
    if (ret >= 0) {
      LOG(ERROR) << "invalid interfacename should fail!";
      close(ret);
    }
  } else if (test_name == "invalid_rxq") {
    struct dma_buf_frags_bind_rx_queue bind_cmd;
    bind_cmd.rxq_idx = 1000;
    strcpy(bind_cmd.ifname, ifname);
    int ret = ioctl(dma_buf_frags_fd, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
    if (ret >= 0) {
      LOG(ERROR) << "Invalid rxqueue should fail!";
      close(ret);
    }
  } else if (test_name == "invalid_dmabuf_fd") {
    struct dma_buf_frags_bind_rx_queue bind_cmd;
    bind_cmd.rxq_idx = 1;
    strcpy(bind_cmd.ifname, ifname);
    int ret = ioctl(dma_buf_frags_fd + 1, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
    if (ret >= 0) {
      LOG(ERROR) << "Invalid dma frag fd should fail!";
      close(ret);
    }
  } else if (test_name == "good") {
    struct dma_buf_frags_bind_rx_queue bind_cmd;
    bind_cmd.rxq_idx = 1;
    strcpy(bind_cmd.ifname, ifname);
    int ret = ioctl(dma_buf_frags_fd, DMA_BUF_FRAGS_BIND_RX, &bind_cmd);
    if (ret < 0) {
      LOG(ERROR) << "Bind queue fail, plz check!";
    } else {
      close(ret);
    }
  } else {
    LOG(INFO) << "No test was run.";
  }

  close(dma_buf_frags_fd);
close_dmabuf_fd:
  close(dma_buf_fd);
close_procfs_fd:
  close(procfs_fd);
out:
  close(fd);
  CU_ASSERT_SUCCESS(cuMemFree(p));

  CUcontext old_ctx;
  CU_ASSERT_SUCCESS(cuCtxPopCurrent(&old_ctx));
  return status;
}
