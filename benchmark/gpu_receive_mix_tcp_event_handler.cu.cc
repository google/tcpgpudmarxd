#include "experimental/users/chechenglin/tcpgpudmad/benchmark/gpu_receive_mix_tcp_event_handler.cu.h"

#include <fcntl.h>
#include <sys/socket.h>

#include <mutex>

#include "experimental/users/chechenglin/tcpgpudmad/benchmark/gpu_receive_common.cu.h"
#include "third_party/absl/strings/str_format.h"

namespace tcpdirect {

GpuReceiveMixTcpEventHandler::GpuReceiveMixTcpEventHandler(
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

  validator_.reset(new ValidationReceiverCtx(message_size));
}

GpuReceiveMixTcpEventHandler::~GpuReceiveMixTcpEventHandler() {}

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
    error_ = absl::StrFormat("recv(error): ret: %d, errno: %d", ret, errno);
  }
}

bool GpuReceiveMixTcpEventHandler::RecvFromSocket() {
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
    error_ = absl::StrFormat("recvmsg(error): ret: %d, errno: %d", ret, errno);
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
    }

    /*
    if (pg_token_found > 0) {
      LOG(INFO) << "Got " << pg_token_found << " tokens";
    }
    */

    size_t host_buf_offset = 0;
    size_t dst_offset = bytes_recv_;
    for (int i = 0; i < rx_blks_.size(); i++) {
      auto& blk = rx_blks_[i];
      size_t off = (size_t)blk.gpu_offset;
      bool is_host = (blk.type == TcpDirectRxBlock::Type::kHost);
      if (is_host) {
        if (blk.size > 0) {
          CU_ASSERT_SUCCESS(cuMemcpyHtoDAsync(
              gpu_rx_mem_ + dst_offset, recv_buf_.data() + host_buf_offset,
              blk.size, stream_));
          host_buf_offset += blk.size;
        }
      } else {
        scattered_data_.emplace_back(
            make_long3((long)dst_offset, (long)off, (long)blk.size));
      }
      dst_offset += blk.size;
      // LOG(INFO) << "blk.size == " << blk.size;
    }

    msg_.msg_control = ctrl_data_;
    msg_.msg_controllen = sizeof(ctrl_data_);
    bytes_recv_ += ret;
  }
  return bytes_recv_ == message_size_;
}

void GpuReceiveMixTcpEventHandler::Reset() {
  FreeRxPages(token_to_free_, socket_);
  rx_blks_.clear();
  token_to_free_.clear();
  scattered_data_.clear();
  bytes_recv_ = 0;
}

double GpuReceiveMixTcpEventHandler::GetRxBytes() {
  double ret = (double)epoch_rx_bytes_;
  epoch_rx_bytes_ = 0;
  return ret;
}
}  // namespace tcpdirect
