#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <fcntl.h>
#include <sys/socket.h>

#include <mutex>

#include "machine_test/cuda/gpu_receive_common.cuh"
#include "machine_test/cuda/gpu_receive_miss_flag_event_handler.cuh"

namespace gpudirect_tcpxd {

bool GpuReceiveMissFlagEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLIN) {
    if (bytes_recv_ == message_size_) {
      CudaStreamSync();
      Reset();
    }
    iov_.iov_base = recv_buf_.data();
    iov_.iov_len = message_size_ - bytes_recv_;
    ssize_t ret = recvmsg(socket_, &msg_, MSG_DONTWAIT);
    if (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      LOG(WARNING) << thread_id_ << ": "
                   << absl::StrFormat("Got EAGAIN | EWOULDBLOCK: %d %x", ret,
                                      errno);
    }
    if (ret < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
      error_msg_ =
          absl::StrFormat("recvmsg(error), ret: %d, errno: %d", ret, errno);
      error_ = false;
    }
    if (ret >= 0) {
      error_msg_ = "recvmsg(no error): it should error out here, please check.";
      error_ = true;
      return false;
    }
  }
  return true;
}

}  // namespace gpudirect_tcpxd
