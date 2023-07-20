#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <fcntl.h>
#include <sys/socket.h>

#include <cstdlib>
#include <ctime>
#include <mutex>

#include "machine_test/cuda/gpu_receive_token_free_event_handler.cuh"

namespace gpudirect_tcpxd {
namespace {

__global__ void scatter_copy_kernel(long3* scatter_list, uint8_t* dst,
                                    uint8_t* src) {
  int block_idx = blockIdx.x;
  long3 blk = scatter_list[block_idx];
  long dst_off = blk.x;
  long src_off = blk.y;
  long sz = blk.z;

  int thread_sz = sz / blockDim.x;
  int rem = sz % blockDim.x;
  bool extra = (threadIdx.x < rem);
  int thread_offset = sz / blockDim.x * threadIdx.x;
  thread_offset += (extra) ? threadIdx.x : rem;

  for (int i = 0; i < thread_sz; i++) {
    dst[dst_off + thread_offset + i] = src[src_off + thread_offset + i];
  }
  if (extra) {
    dst[dst_off + thread_offset + thread_sz] =
        src[src_off + thread_offset + thread_sz];
  }
}
}  // namespace

bool GpuReceiveTokenFreeEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLIN) {
    if (bytes_recv_ == message_size_) {
      CudaStreamSync();
      CustomizedReset();
    }
    bool done = CustomizedRecvFromSocket();
    if (HasError()) {
      return false;
    }
    if (done) {
      epoch_rx_bytes_ += message_size_;
    }
  }
  return true;
}

bool GpuReceiveTokenFreeEventHandler::CustomizedRecvFromSocket() {
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

void GpuReceiveTokenFreeEventHandler::CustomizedReset() {
  if (token_to_free_.size() > 0) {
    int ret;
    ret = FreeRxPages(token_to_free_, socket_);
    ret = FreeRxPages(token_to_free_, socket_);
    PLOG(INFO) << "Attempt double free: ret [" << ret << "]:";

    std::vector<TokenT> garbage_tokens;
    // add two random tokens that are invalid
    while (garbage_tokens.size() < 2) {
      uint32_t r = static_cast<uint32_t>(std::rand());
      if (token_to_free_set_.find(r) != token_to_free_set_.end()) continue;
      garbage_tokens.push_back({r, 1});
    }
    ret = FreeRxPages(garbage_tokens, socket_);
    PLOG(INFO) << "Attempt free random tokens: ret [" << ret << "]:";
    // add one token that is semi-legit
    garbage_tokens.clear();
    uint32_t r = token_to_free_.back().token_start + 1;
    while (token_to_free_set_.find(r) != token_to_free_set_.end()) {
      ++r;
    }
    garbage_tokens.push_back({r, 1});
    ret = FreeRxPages(garbage_tokens, socket_);
    PLOG(INFO) << "Attempt free semi-legit tokens: ret [" << ret << "]:";
    // use blks after they are "freed"
    size_t dst_offset = bytes_recv_;
    for (int i = 0; i < rx_blks_.size(); i++) {
      auto& blk = rx_blks_[i];
      size_t off = (size_t)blk.gpu_offset;
      bool is_host = (blk.type == TcpDirectRxBlock::Type::kHost);
      if (is_host) {
        continue;
      } else {
        scattered_data_.emplace_back(
            make_long3((long)dst_offset, (long)off, (long)blk.size));
      }
      dst_offset += blk.size;
      // LOG(INFO) << "blk.size == " << blk.size;
    }
    CU_ASSERT_SUCCESS(
        cuMemcpyHtoDAsync(gpu_scatter_list_, scattered_data_.data(),
                          scattered_data_.size() * sizeof(long3), stream_));

    scatter_copy_kernel<<<scattered_data_.size(), 256, 0, stream_>>>(
        (long3*)gpu_scatter_list_, (uint8_t*)gpu_rx_mem_,
        (uint8_t*)gpu_page_handle_->GetGpuMem());
    LOG(INFO)
        << "Attempt reference gpu memory after corresponding tokens are freed";
  }

  rx_blks_.clear();
  token_to_free_.clear();
  token_to_free_set_.clear();
  scattered_data_.clear();
  bytes_recv_ = 0;
}
}  // namespace gpudirect_tcpxd
