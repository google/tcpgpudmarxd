#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <fcntl.h>
#include <sys/socket.h>

#include <mutex>

#include "machine_test/cuda/gpu_receive_common.cuh"
#include "machine_test/cuda/gpu_receive_mix_tcp_event_handler.cuh"

namespace gpudirect_tcpxd {

bool GpuReceiveMixTcpEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLIN) {
    if (bytes_recv_ == message_size_) {
      CudaStreamSync();
      Reset();
    }
    if (use_tcp_) {
      TcpRecvFromSocket();
      use_tcp_ = false;
    } else {
      RecvFromSocket();
      use_tcp_ = true;
    }
    if (HasError()) {
      return false;
    }
  }
  return true;
}

void GpuReceiveMixTcpEventHandler::TcpRecvFromSocket() {
  ssize_t ret = recv(socket_, &recv_buf_.data()[bytes_recv_],
                     message_size_ - bytes_recv_, MSG_DONTWAIT);
  if (ret < 0 && (errno != EAGAIN && errno != EWOULDBLOCK)) {
    error_msg_ = absl::StrFormat("recv(error): ret: %d, errno: %d", ret, errno);
    error_ = true;
  }
}
}  // namespace gpudirect_tcpxd
