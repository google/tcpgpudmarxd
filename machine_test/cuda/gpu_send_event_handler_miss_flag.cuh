#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_SEND_EVENT_HANDLER_MISS_FLAG_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_SEND_EVENT_HANDLER_MISS_FLAG_H_

#include <cuda.h>
#include <linux/errqueue.h>
#include <linux/types.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/socket.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "cuda/gpu_page_allocator_interface.cuh"
#include "machine_test/cuda/gpu_send_event_handler.cuh"

namespace gpudirect_tcpxd {
class GpuSendEventHandlerMissFlag : public GpuSendEventHandler {
 public:
  GpuSendEventHandlerMissFlag(
      std::string thread_id, int socket, size_t message_size,
      bool do_validation,
      std::unique_ptr<GpuPageAllocatorInterface> gpu_page_allocator)
      : GpuSendEventHandler(thread_id, socket, message_size, do_validation,
                            std::move(gpu_page_allocator)) {}
  bool HandleEvents(unsigned events) override;

 private:
  bool CustomizedHandleEPollOut();
};

}  // namespace gpudirect_tcpxd
#endif
