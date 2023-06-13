#include "experimental/users/chechenglin/tcpgpudmad/benchmark/gpu_receive_no_token_free_event_handler.cu.h"

#include <fcntl.h>
#include <sys/socket.h>

#include <cstdlib>
#include <ctime>
#include <mutex>

#include "third_party/absl/strings/str_format.h"

namespace tcpdirect {

GpuReceiveNoTokenFreeEventHandler::GpuReceiveNoTokenFreeEventHandler(
    std::string thread_id, int socket, size_t message_size, bool do_validation,
    std::string gpu_pci_addr) {
  std::srand(std::time(nullptr));
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

GpuReceiveNoTokenFreeEventHandler::~GpuReceiveNoTokenFreeEventHandler() {}

bool GpuReceiveNoTokenFreeEventHandler::HandleEvents(unsigned events) {
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

bool GpuReceiveNoTokenFreeEventHandler::RecvFromSocket() {
  iov_.iov_base = recv_buf_.data();
  iov_.iov_len = message_size_ - bytes_recv_;
  ssize_t ret = recvmsg(socket_, &msg_, MSG_SOCK_DEVMEM | MSG_DONTWAIT);
  if (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    LOG(WARNING) << thread_id_ << ": "
                 << absl::StrFormat("Got EAGAIN | EWOULDBLOCK: %d %x", ret,
                                    errno);
  }
  if (ret < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    // LOG(ERROR) << "recvmsg error: ";
    error_ = true;
    return false;
  }
  if (ret >= 0) {
    struct cmsghdr* cm = nullptr;
    struct devmemvec* devmemvec = nullptr;
    int off_found = 0;
    int pg_token_found = 0;
    rx_blks_.clear();
    for (cm = CMSG_FIRSTHDR(&msg_); cm; cm = CMSG_NXTHDR(&msg_, cm)) {
      if (cm->cmsg_level != SOL_SOCKET ||
          (cm->cmsg_type != SCM_DEVMEM_OFFSET &&
           cm->cmsg_type != SCM_DEVMEM_HEADER)) {
        LOG(ERROR) << thread_id_ << ": "
                   << absl::StrFormat("cmsg: unknown %u.%u\n", cm->cmsg_level,
                                      cm->cmsg_type);
        continue;
      }

      CHECK(off_found >= pg_token_found && off_found - pg_token_found <= 1);

      devmemvec = (struct devmemvec*)CMSG_DATA(cm);

      if (cm->cmsg_type == SCM_DEVMEM_HEADER) {
        // TODO: process data copied from skb's linear buffer
        TcpDirectRxBlock blk;
        blk.type = TcpDirectRxBlock::Type::kHost;
        blk.size = devmemvec->frag_size;
        rx_blks_.emplace_back(blk);
        continue;
      }

      /* current version returns two cmsgs:
       * - one with offset from start of region
       * - one with raw physaddr */
      TcpDirectRxBlock blk;
      blk.type = TcpDirectRxBlock::Type::kGpu;
      blk.gpu_offset = (uint64_t)devmemvec->frag_offset;
      blk.size = devmemvec->frag_size;
      rx_blks_.emplace_back(blk);
      off_found++;
      TokenT pg_token = {devmemvec->frag_token & ~(1 << 31), 1};
      pg_token_found++;
      token_to_free_.push_back(pg_token);
      token_to_free_set_.insert(pg_token.token_start);
    }

    /*
    if (pg_token_found > 0) {
      LOG(INFO) << "Got " << pg_token_found << " tokens";
    }
    */

    msg_.msg_control = ctrl_data_;
    msg_.msg_controllen = sizeof(ctrl_data_);
    bytes_recv_ += ret;
  }
  return bytes_recv_ == message_size_;
}

// Don't SO_DEVMEM_DONTNEED to test not freeing RX buffer entries.
void GpuReceiveNoTokenFreeEventHandler::Reset() {}

double GpuReceiveNoTokenFreeEventHandler::GetRxBytes() {
  double ret = (double)epoch_rx_bytes_;
  epoch_rx_bytes_ = 0;
  return ret;
}
}  // namespace tcpdirect
