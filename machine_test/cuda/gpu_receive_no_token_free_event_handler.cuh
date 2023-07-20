#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPDIRECT_BENCH_GPU_RECEIVE_NO_TOKEN_FREE_EVENT_HANDLER_CU_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPDIRECT_BENCH_GPU_RECEIVE_NO_TOKEN_FREE_EVENT_HANDLER_CU_H_

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

class GpuReceiveNoTokenFreeEventHandler : public GpuReceiveEventHandler {
 public:
  GpuReceiveNoTokenFreeEventHandler(std::string thread_id, int socket,
                                    size_t message_size, bool do_validation,
                                    std::string gpu_pci_addr)
      : GpuReceiveEventHandler(thread_id, socket, message_size, do_validation,
                               gpu_pci_addr) {}
  virtual bool HandleEvents(unsigned events) override;
};

}  // namespace gpudirect_tcpxd
#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPDIRECT_BENCH_GPU_RECEIVE_NO_TOKEN_FREE_EVENT_HANDLER_CU_H_
