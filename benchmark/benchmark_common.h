#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_BENCHMARK_COMMON_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_BENCHMARK_COMMON_H_

#include <pthread.h>

#include <atomic>
#include <string>
#include <unordered_map>

#include "experimental/users/chechenglin/tcpgpudmad/benchmark/socket_helper.h"

namespace tcpdirect {
#define MSG_SOCK_DEVMEM 0x2000000

using NicPciAddrToIpMap = std::unordered_map<std::string, union SocketAddress>;
using Dictionary = std::unordered_map<std::string, std::string>;

struct ThreadId {
  int gpu_idx;
  int per_gpu_thread_idx;
};

void GetNicPciAddrToIpMap(NicPciAddrToIpMap* nic_pci_addr_to_ip_addr,
                          const Dictionary& pci_to_ifname);

union SocketAddress GetIpv6FromIfName(const std::string& ifname);

}  // namespace tcpdirect
#endif  // _THIRD_PARTY_TCPDIRECT_RX_MAANGER_BENCH_BENCHMARK_COMMON_H_
