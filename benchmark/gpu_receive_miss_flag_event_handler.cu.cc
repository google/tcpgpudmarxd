#include "experimental/users/chechenglin/tcpgpudmad/benchmark/gpu_receive_miss_flag_event_handler.cu.h"

#include <fcntl.h>
#include <sys/socket.h>

#include <mutex>

#include "experimental/users/chechenglin/tcpgpudmad/benchmark/gpu_receive_common.cu.h"
#include "third_party/absl/strings/str_format.h"

namespace tcpdirect {

GpuReceiveMissFlagEventHandler::GpuReceiveMissFlagEventHandler(
    std::string thread_id, int socket, size_t message_size, bool do_validation,
    std::string gpu_pci_addr) {
  thread_id_ = thread_id;
  socket_ = socket;
  message_size_ = message_size;
  CUstream stream;
  CU_ASSERT_SUCCESS(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  debug_ = true;
  rx_buff_ = gpumem_import(gpu_pci_addr, thread_id_);
  stream_ = stream;

  memset(&msg_, 0, sizeof(msg_));
  msg_.msg_control = ctrl_data_;
  msg_.msg_controllen = sizeof(ctrl_data_);
  msg_.msg_iov = &iov_;
  msg_.msg_iovlen = 1;
  message_size_ = message_size;

  recv_buf_.resize(message_size);
  bytes_recv_ = 0;

  CU_ASSERT_SUCCESS(cuMemAlloc(&gpu_rx_mem_, message_size));
  CU_ASSERT_SUCCESS(cuMemAlloc(&gpu_scatter_list_, message_size));

  error_ = false;
}

GpuReceiveMissFlagEventHandler::~GpuReceiveMissFlagEventHandler() {}

bool GpuReceiveMissFlagEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLIN) {
    if (bytes_recv_ == message_size_) {
      CudaStreamSync();
      Reset();
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

bool GpuReceiveMissFlagEventHandler::RecvFromSocket() {
  iov_.iov_base = recv_buf_.data();
  iov_.iov_len = message_size_ - bytes_recv_;
  ssize_t ret = recvmsg(socket_, &msg_, MSG_DONTWAIT);
  if (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    LOG(WARNING) << thread_id_ << ": "
                 << absl::StrFormat("Got EAGAIN | EWOULDBLOCK: %d %x", ret,
                                    errno);
  }
  if (ret < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    error_ = absl::StrFormat("recvmsg(error), ret: %d, errno: %d", ret, errno);
  }
  if (ret >= 0) {
    error_ = "recvmsg(no error): it should error out here, please check.";
  }
  return false;
}

void GpuReceiveMissFlagEventHandler::Reset() { bytes_recv_ = 0; }

double GpuReceiveMissFlagEventHandler::GetRxBytes() {
  double ret = (double)epoch_rx_bytes_;
  epoch_rx_bytes_ = 0;
  return ret;
}
}  // namespace tcpdirect
