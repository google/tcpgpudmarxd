#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_GPU_PAGE_EXPORTER_INTERFACE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_GPU_PAGE_EXPORTER_INTERFACE_H_

#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <net/if.h>
#include <sys/ioctl.h>

#include <memory>
#include <string>
#include <vector>

#include "experimental/users/chechenglin/tcpgpudmad/proto/gpu_rxq_configuration.proto.h"
#include "third_party/absl/status/status.h"

namespace tcpdirect {

#define DMA_BUF_BASE 'b'
#define RX_POOL_SIZE (1UL << 32)
#define DMA_BUF_FRAGS_BIND_RX \
  _IOW(DMA_BUF_BASE, 3, struct dma_buf_frags_bind_rx_queue)

struct dma_buf_frags_bind_rx_queue {
  char ifname[IFNAMSIZ];
  __u32 rxq_idx;
};

class GpuPageExporterInterface {
 public:
  virtual ~GpuPageExporterInterface() = default;
  virtual absl::Status Initialize(const GpuRxqConfigurationList& config_list,
                                  const std::string& prefix) = 0;
  virtual absl::Status Export() = 0;
  virtual void Cleanup() = 0;
  static int gpumem_bind_rxq(int fd, const std::string& ifname, int rxqid);
};
}  // namespace tcpdirect
#endif
