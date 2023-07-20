#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_TCP_SEND_EVENT_HANDLER_SEND_TCP_DIRECT_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_TCP_SEND_EVENT_HANDLER_SEND_TCP_DIRECT_H_

#include <poll.h>
#include <sys/epoll.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

#include "machine_test/cuda/validation.cuh"
#include "machine_test/include/event_handler_interface.h"

namespace gpudirect_tcpxd {
class TcpSendTcpDirectEventHandler : public EventHandlerInterface {
 public:
  TcpSendTcpDirectEventHandler(std::string thread_id, int socket,
                               size_t message_size, bool do_validation);
  ~TcpSendTcpDirectEventHandler() override = default;
  unsigned InterestedEvents() override { return EPOLLOUT | EPOLLERR; }
  bool HandleEvents(unsigned events) override;
  double GetRxBytes() override { return 0.0; }
  double GetTxBytes() override;
  std::string Error() override { return ""; }

 private:
  bool HandleEPollOut();
  bool HandleEPollErr();

  std::string thread_id_;
  int socket_;
  size_t message_size_;
  bool do_validation_;
  std::unique_ptr<char[]> tx_buf_;
  size_t tx_offset_;
  std::atomic<uint64_t> epoch_tx_bytes_{0};
  std::unique_ptr<ValidationSenderCtx> validator_;
};

}  // namespace gpudirect_tcpxd

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPDIRECT_BENCH_TCP_SEND_EVENT_HANDLER_SEND_TCP_DIRECT_CU_H_
