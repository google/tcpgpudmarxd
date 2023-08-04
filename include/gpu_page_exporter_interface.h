/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_GPU_PAGE_EXPORTER_INTERFACE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_GPU_PAGE_EXPORTER_INTERFACE_H_

#include <absl/status/status.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <net/if.h>
#include <sys/ioctl.h>

#include <memory>
#include <string>
#include <vector>

#include "proto/gpu_rxq_configuration.pb.h"

namespace gpudirect_tcpxd {

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
}  // namespace gpudirect_tcpxd
#endif
