#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_SEND_EVENT_HANDLER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_SEND_EVENT_HANDLER_H_

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
#include "machine_test/cuda/validation.cuh"
#include "machine_test/include/event_handler_interface.h"

namespace gpudirect_tcpxd {
class GpuSendEventHandler : public EventHandlerInterface {
 public:
  GpuSendEventHandler(
      std::string thread_id, int socket, size_t message_size,
      bool do_validation,
      std::unique_ptr<GpuPageAllocatorInterface> gpu_page_allocator);
  ~GpuSendEventHandler() = default;
  unsigned InterestedEvents() override { return EPOLLOUT | EPOLLERR; }
  bool HandleEvents(unsigned events) override;
  double GetRxBytes() override { return 0.0; }
  double GetTxBytes() override;
  std::string Error() override { return error_; }

 protected:
  bool HandleEPollOut();
  bool HandleEPollErr();

  bool HasError() { return !error_.empty(); }

  CUdeviceptr GetTxGpuMem() { return gpu_page_allocator_->GetGpuMem(0); }

  bool PendingSendDone() {
    // query error queue of the socket, check if all zero copy pages could be
    // reused
    return sendmsg_cnt_ == 0 || sendmsg_last_ == sendmsg_cnt_ - 1;
  }

  void Reset();

  std::string thread_id_;
  size_t message_size_;
  bool do_validation_;
  size_t tx_offset_;
  std::atomic<long> epoch_tx_bytes_{0};

  std::unique_ptr<char[]> buf_;
  int gpu_mem_fd_;
  int dma_buf_fd_;
  bool use_dmabuf_;

  int sendmsg_cnt_;
  int sendmsg_last_;
  int socket_;

  std::string error_;

  struct msghdr msg_;
  char ctrl_data_[CMSG_SPACE(sizeof(int) * 2)];
  size_t bytes_sent_;
  struct iovec iov_;

  struct msghdr errq_msg_;
  char err_ctrl_data_[100 * CMSG_SPACE(sizeof(int))];
  std::unique_ptr<GpuPageAllocatorInterface> gpu_page_allocator_;
  std::unique_ptr<ValidationSenderCtx> validator_;
  unsigned long msg_id_;
};

}  // namespace gpudirect_tcpxd
#endif
