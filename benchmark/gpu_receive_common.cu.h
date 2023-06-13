#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_COMMON_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_COMMON_H_

#include <linux/types.h>

#include <string>
#include <vector>

#include "experimental/users/chechenglin/tcpgpudmad/benchmark/tcpdirect_common.h"
#include "experimental/users/chechenglin/tcpgpudmad/cuda/gpu_page_handle_interface.cu.h"

namespace tcpdirect {

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
}  // namespace tcpdirect
#endif
