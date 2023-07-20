#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_MISS_FLAG_EVENT_HANDLER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_MISS_FLAG_EVENT_HANDLER_H_

#include <cuda.h>
#include <poll.h>
#include <sys/epoll.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuda/common.cuh"
#include "machine_test/cuda/gpu_receive_event_handler.cuh"
#include "machine_test/include/benchmark_common.h"

namespace gpudirect_tcpxd {

class GpuReceiveMissFlagEventHandler : public GpuReceiveEventHandler {
 public:
  GpuReceiveMissFlagEventHandler(std::string thread_id, int socket,
                                 size_t message_size, bool do_validation,
                                 std::string gpu_pci_addr)
      : GpuReceiveEventHandler(thread_id, socket, message_size, do_validation,
                               gpu_pci_addr) {}
  virtual bool HandleEvents(unsigned events) override;
  std::string Error() override { return error_msg_; }

 private:
  std::string error_msg_;
};

}  // namespace gpudirect_tcpxd
#endif
