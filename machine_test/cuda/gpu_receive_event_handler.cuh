#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_EVENT_HANDLER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_GPU_RECEIVE_EVENT_HANDLER_H_

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
#include "cuda/cu_ipc_memfd_handle_importer.cuh"
#include "machine_test/cuda/gpu_receive_common.cuh"
#include "machine_test/cuda/validation.cuh"
#include "machine_test/include/benchmark_common.h"
#include "machine_test/include/event_handler_interface.h"

namespace gpudirect_tcpxd {

class GpuReceiveEventHandler : public EventHandlerInterface {
 public:
  GpuReceiveEventHandler(std::string thread_id, int socket, size_t message_size,
                         bool do_validation, std::string gpu_pci_addr);
  ~GpuReceiveEventHandler() override;
  virtual unsigned InterestedEvents() override { return EPOLLIN; }
  virtual bool HandleEvents(unsigned events) override;
  virtual double GetRxBytes() override;
  virtual double GetTxBytes() override { return 0.0; }
  std::string Error() override { return ""; }

 protected:
  bool HasError() { return error_; }
  CUdeviceptr GetRxGpuMem() { return gpu_rx_mem_; }  // for validation
  CUstream GetCudaStream() { return stream_; }       // for validation
  void CudaStreamSync() { CU_ASSERT_SUCCESS(cuStreamSynchronize(stream_)); }
  bool RecvFromSocket();
  void GatherRxData();
  void Reset();

  std::string thread_id_;
  int socket_;
  size_t message_size_;
  size_t rx_offset_;
  std::atomic<long> epoch_rx_bytes_{0};

  std::unique_ptr<GpuPageHandleInterface> gpu_page_handle_;
  CUdeviceptr gpu_rx_mem_;
  CUdeviceptr gpu_scatter_list_;
  CUstream stream_;

  bool error_;
  bool debug_;

  struct msghdr msg_;
  struct iovec iov_;
  char ctrl_data_[10000 * CMSG_SPACE(sizeof(struct iovec))];

  size_t bytes_recv_;

  std::vector<uint8_t> recv_buf_;
  std::vector<TcpDirectRxBlock> rx_blks_;
  std::vector<TokenT> token_to_free_;

  std::vector<long3> scattered_data_;
  bool do_validation_;
  std::unique_ptr<ValidationReceiverCtx> validator_;
};

}  // namespace gpudirect_tcpxd
#endif
