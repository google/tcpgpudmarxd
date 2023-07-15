// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/gpu_page_exporter_interface.h"

#include <string>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

namespace tcpdirect {
int GpuPageExporterInterface::gpumem_bind_rxq(int fd, const std::string& ifname,
                                              int rxqid) {
  struct dma_buf_frags_bind_rx_queue bind_cmd;
  strcpy(bind_cmd.ifname, ifname.c_str());
  bind_cmd.rxq_idx = rxqid;
  if (int ret = ioctl(fd, DMA_BUF_FRAGS_BIND_RX, &bind_cmd); ret < 0) {
    LOG(INFO) << absl::StrFormat(
        "failed to bind queue for %s, queue id: %d, error: %s", ifname.c_str(),
        rxqid, strerror(errno));
    return ret;
  }
  return 0;
}
}  // namespace tcpdirect
