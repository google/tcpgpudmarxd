#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_DOUBLE_FREE_EVENT_HANDLER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_DOUBLE_FREE_EVENT_HANDLER_H_

#include <cuda.h>
#include <poll.h>
#include <sys/epoll.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "cuda/common.cuh"
#include "machine_test/cuda/gpu_receive_common.cuh"
#include "machine_test/cuda/gpu_receive_event_handler.cuh"
#include "machine_test/include/benchmark_common.h"

namespace gpudirect_tcpxd {

class GpuReceiveTokenFreeEventHandler : public GpuReceiveEventHandler {
 public:
  GpuReceiveTokenFreeEventHandler(std::string thread_id, int socket,
                                  size_t message_size, bool do_validation,
                                  std::string gpu_pci_addr)
      : GpuReceiveEventHandler(thread_id, socket, message_size, do_validation,
                               gpu_pci_addr) {}
  virtual bool HandleEvents(unsigned events) override;

 private:
  std::unordered_set<uint32_t> token_to_free_set_;
  bool CustomizedRecvFromSocket();
  void CustomizedReset();
};

}  // namespace gpudirect_tcpxd
#endif
