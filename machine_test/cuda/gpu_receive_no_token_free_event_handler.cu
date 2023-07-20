#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <fcntl.h>
#include <sys/socket.h>

#include <cstdlib>
#include <ctime>
#include <mutex>

#include "machine_test/cuda/gpu_receive_no_token_free_event_handler.cuh"

namespace gpudirect_tcpxd {

bool GpuReceiveNoTokenFreeEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLIN) {
    if (bytes_recv_ == message_size_) {
      CudaStreamSync();
      // no reset
    }
    bool done = RecvFromSocket();
    if (HasError()) {
      return false;
    }
    if (done) {
      epoch_rx_bytes_ += message_size_;
    }
  }
  return true;
}

}  // namespace gpudirect_tcpxd
