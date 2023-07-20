#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_COMMON_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_COMMON_H_

#include <linux/types.h>

#include <string>
#include <vector>

#include "cuda/gpu_page_handle_interface.cuh"
#include "machine_test/include/tcpdirect_common.h"

namespace gpudirect_tcpxd {

struct TcpDirectRxBlock {
  enum Type {
    kHost,
    kGpu,
  };
  Type type;
  uint64_t gpu_offset;
  size_t size;
  uint64_t paddr;
};

struct devmemvec {
  __u32 frag_offset;
  __u32 frag_size;
  __u32 frag_token;
};

struct devmemtoken {
  __u32 token_start;
  __u32 token_count;
};

using TokenT = devmemtoken;

CUdeviceptr gpumem_import(const std::string& gpu_pci_addr,
                          const std::string& thread_id);
int FreeRxPages(const std::vector<TokenT>& tokens_to_free, int socket);
}  // namespace gpudirect_tcpxd
#endif
