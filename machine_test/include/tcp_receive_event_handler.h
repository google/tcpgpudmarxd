#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_TCP_RECEIVE_EVENT_HANDLER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_TCP_RECEIVE_EVENT_HANDLER_H_

#include <poll.h>
#include <sys/epoll.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

#include "machine_test/cuda/validation.cuh"
#include "machine_test/include/event_handler_interface.h"

namespace gpudirect_tcpxd {
class TcpReceiveEventHandler : public EventHandlerInterface {
 public:
  TcpReceiveEventHandler(std::string thread_id, int socket, size_t message_size,
                         bool do_validation);
  ~TcpReceiveEventHandler() override = default;
  unsigned InterestedEvents() override { return EPOLLIN; }
  bool HandleEvents(unsigned events) override;
  double GetRxBytes() override;
  double GetTxBytes() override { return 0.0; };
  std::string Error() override { return ""; }

 private:
  std::string thread_id_;
  int socket_;
  size_t message_size_;
  bool do_validation_;
  std::unique_ptr<char[]> rx_buf_;
  size_t rx_offset_;
  std::atomic<uint64_t> epoch_rx_bytes_{0};
  std::unique_ptr<ValidationReceiverCtx> validator_;
};

}  // namespace gpudirect_tcpxd
#endif
