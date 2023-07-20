#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "cuda/common.cuh"
#include "machine_test/cuda/gpu_send_oob_event_handler.cuh"
#include "machine_test/include/benchmark_common.h"
#include "machine_test/include/tcpdirect_common.h"

namespace gpudirect_tcpxd {

bool GpuSendOobEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLOUT) {
    if (!CustomizedHandleEPollOut()) return false;
  }
  if (events & EPOLLERR) {
    if (!HandleEPollErr()) return false;
  }
  return true;
}

bool GpuSendOobEventHandler::CustomizedHandleEPollOut() {
  if (bytes_sent_ == message_size_) {
    if (!PendingSendDone()) {
      return true;
    }
    epoch_tx_bytes_ += message_size_;
    Reset();
  }
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg_);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_DEVMEM_OFFSET;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int) * 2);
  *((int *)CMSG_DATA(cmsg)) = gpu_page_allocator_->GetGpuMemFd(msg_id_);
  ((int *)CMSG_DATA(cmsg))[1] = (int)message_size_ + 1;

  iov_.iov_base = &(buf_.get())[bytes_sent_];
  iov_.iov_len = message_size_ - bytes_sent_;
  ssize_t ret =
      sendmsg(socket_, &msg_, MSG_ZEROCOPY | MSG_SOCK_DEVMEM | MSG_DONTWAIT);
  if (ret < 0 && errno != EWOULDBLOCK && errno != EAGAIN) {
    error_ =
        absl::StrFormat("sendmsg() error,  ret: %d, errno: %d", ret, errno);
    // PLOG(ERROR) << "sendmsg() error: ";
    return false;
  }
  bytes_sent_ += ret;
  sendmsg_cnt_++;

  if (HasError()) {
    return false;
  }
  return true;
}
}  // namespace gpudirect_tcpxd
