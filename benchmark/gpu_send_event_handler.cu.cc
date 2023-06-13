#include "experimental/users/chechenglin/tcpgpudmad/benchmark/gpu_send_event_handler.cu.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "experimental/users/chechenglin/tcpgpudmad/benchmark/benchmark_common.h"
#include "experimental/users/chechenglin/tcpgpudmad/benchmark/tcpdirect_common.h"
#include "experimental/users/chechenglin/tcpgpudmad/cuda/common.cu.h"
#include "third_party/absl/strings/str_format.h"

namespace tcpdirect {

namespace {
#define GPUMEM_ALIGNMENT (1UL << 21)
#define GPUMEM_MINSZ 0x400000

__device__ void compute_range(size_t sz, size_t dim, size_t idx, size_t &offset,
                              size_t &range_sz) {
  size_t chunk_sz = sz / dim;
  size_t rem = sz % dim;
  bool extra = (idx < rem);
  offset = chunk_sz * idx;
  offset += (extra) ? idx : rem;
  range_sz = chunk_sz;
  range_sz += (extra) ? 1 : 0;
}

__global__ void data_update_kernel(uint32_t *data, size_t num_ele,
                                   uint32_t inc_step) {
  size_t blk_offset, blk_sz;
  size_t thread_offset, thread_sz;
  compute_range(num_ele, gridDim.x, blockIdx.x, blk_offset, blk_sz);
  compute_range(blk_sz, blockDim.x, threadIdx.x, thread_offset, thread_sz);
  size_t offset = blk_offset + thread_offset;
  for (size_t i = 0; i < thread_sz; i++) {
    data[offset + i] += inc_step;
  }
}
}  // namespace

void GpuSendEventHandler::Reset() {
  bytes_sent_ = 0;
  gpu_page_allocator_->Reset();
}

GpuSendEventHandler::GpuSendEventHandler(
    std::string thread_id, int socket, size_t message_size, bool do_validation,
    std::unique_ptr<GpuPageAllocatorInterface> gpu_page_allocator) {
  thread_id_ = thread_id;
  message_size_ = message_size;
  do_validation_ = do_validation;

  buf_.reset(new char[message_size_]);
  gpu_page_allocator_ = std::move(gpu_page_allocator);
  bool success;
  gpu_page_allocator_->AllocatePage(message_size_, &msg_id_, &success);

  socket_ = socket;
  sendmsg_cnt_ = 0;
  sendmsg_last_ = 0xffffffff;

  memset(&msg_, 0, sizeof(msg_));
  memset(ctrl_data_, 0, sizeof(ctrl_data_));
  msg_.msg_iov = &iov_;
  msg_.msg_iovlen = 1;
  msg_.msg_control = ctrl_data_;
  msg_.msg_controllen = sizeof(ctrl_data_);
  bytes_sent_ = 0;

  memset(&errq_msg_, 0, sizeof(errq_msg_));
  errq_msg_.msg_control = err_ctrl_data_;
  errq_msg_.msg_controllen = sizeof(err_ctrl_data_);

  if (do_validation_) {
    validator_.reset(new ValidationSenderCtx(message_size_));
    validator_->InitSender([this](uint32_t *arr, int num_u32) {
      CU_ASSERT_SUCCESS(
          cuMemcpyHtoD(GetTxGpuMem(), arr, sizeof(uint32_t) * num_u32));
      CU_ASSERT_SUCCESS(cuCtxSynchronize());
    });
  }
}

bool GpuSendEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLOUT) {
    if (!HandleEPollOut()) return false;
  }
  if (events & EPOLLERR) {
    if (!HandleEPollErr()) return false;
  }
  return true;
}

bool GpuSendEventHandler::HandleEPollOut() {
  if (bytes_sent_ == message_size_) {
    if (!PendingSendDone()) {
      return true;
    }
    epoch_tx_bytes_ += message_size_;
    if (do_validation_) {
      validator_->UpdateSender([this](int num_u32, int send_inc_step) {
        data_update_kernel<<<1, 256>>>((uint32_t *)GetTxGpuMem(), num_u32,
                                       send_inc_step);
        CU_ASSERT_SUCCESS(cuCtxSynchronize());
      });
    }
    Reset();
  }
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg_);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_DEVMEM_OFFSET;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int) * 2);
  *((int *)CMSG_DATA(cmsg)) = gpu_page_allocator_->GetGpuMemFd(msg_id_);
  ((int *)CMSG_DATA(cmsg))[1] = (int)bytes_sent_;

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

bool GpuSendEventHandler::HandleEPollErr() {
  errq_msg_.msg_control = err_ctrl_data_;
  errq_msg_.msg_controllen = sizeof(err_ctrl_data_);
  int ret = recvmsg(socket_, &errq_msg_, MSG_ERRQUEUE);
  if (ret < 0) return true;  // return false?
  // PCHECK(ret >= 0);
  struct sock_extended_err *serr;
  struct cmsghdr *cm;

  cm = CMSG_FIRSTHDR(&errq_msg_);
  while (cm) {
    if (cm->cmsg_level != SOL_IP && cm->cmsg_level != SOL_IPV6 &&
        cm->cmsg_type != IP_RECVERR)
      LOG(FATAL) << thread_id_ << ": cmsg";

    serr = (struct sock_extended_err *)CMSG_DATA(cm);
    if (serr->ee_errno != 0 || serr->ee_origin != SO_EE_ORIGIN_ZEROCOPY)
      LOG(FATAL) << thread_id_ << ": serr";
    int hi = serr->ee_data;
    int lo = serr->ee_info;
    if (sendmsg_last_ != lo - 1) {
      LOG(INFO) << thread_id_ << ": "
                << absl::StrFormat("zerocopy completion: %d --> %d", lo, hi);
    }
    CHECK(sendmsg_last_ == lo - 1)
        << absl::StrFormat("zerocopy completion: %d --> %d", lo, hi);
    sendmsg_last_ = hi;
    cm = CMSG_NXTHDR(&errq_msg_, cm);
  }
  return true;
}

double GpuSendEventHandler::GetTxBytes() {
  double ret = (double)epoch_tx_bytes_;
  epoch_tx_bytes_ = 0;
  return ret;
}
}  // namespace tcpdirect
