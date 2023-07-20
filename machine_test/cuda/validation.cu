#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <random>
#include <string>

#include "cuda/common.cuh"
#include "machine_test/cuda/validation.cuh"

namespace gpudirect_tcpxd {
//// ValidationSenderCtx /////

ValidationSenderCtx::ValidationSenderCtx(size_t message_size) {
  num_u32_ = message_size / sizeof(uint32_t);
  iter_cnt_ = 0;
  CU_ASSERT_SUCCESS(
      cuMemAllocHost((void**)&iter0_data_, num_u32_ * sizeof(uint32_t)));
}

ValidationSenderCtx::~ValidationSenderCtx() {
  CU_ASSERT_SUCCESS(cuMemFreeHost((void**)&iter0_data_));
}

void ValidationSenderCtx::InitSender(
    const std::function<void(uint32_t*, int)>& commit_buffer) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> distrib;
  std::uniform_int_distribution<uint32_t> step_distrib(0, 10);
  uint32_t seed = distrib(gen);
  send_inc_step_ = step_distrib(gen);
  uint32_t* arr = iter0_data_;
  arr[0] = send_inc_step_;
  for (int j = 1; j < num_u32_; j++) {
    arr[j] = seed + j;
  }
  commit_buffer(arr, num_u32_);
}

void ValidationSenderCtx::UpdateSender(
    const std::function<void(int, int)>& update_buffer) {
  update_buffer(num_u32_, send_inc_step_);
  iter_cnt_++;
}

ValidationReceiverCtx::ValidationReceiverCtx(size_t message_size) {
  num_u32_ = message_size / sizeof(uint32_t);
  iter_cnt_ = 0;
  CU_ASSERT_SUCCESS(
      cuMemAllocHost((void**)&iter0_data_, num_u32_ * sizeof(uint32_t)));
  CU_ASSERT_SUCCESS(
      cuMemAllocHost((void**)&validation_buf_, num_u32_ * sizeof(uint32_t)));
}

ValidationReceiverCtx::~ValidationReceiverCtx() {
  cuMemFreeHost((void**)&iter0_data_);
  cuMemFreeHost((void**)&validation_buf_);
}

void ValidationReceiverCtx::InitRxBuf(
    const std::function<void(uint32_t*)>& init_buffer) {
  init_buffer(iter0_data_);
  send_inc_step_ = iter0_data_[0];
}

bool ValidationReceiverCtx::ValidateRxData(
    const std::function<void(uint32_t*, int)>& init_buffer,
    const std::function<void(uint32_t*, int)>& copy_buf) {
  uint32_t* ref_arr = iter0_data_;
  if (iter_cnt_ == 0) {
    init_buffer(ref_arr, num_u32_);
    send_inc_step_ = ref_arr[0];
    iter_cnt_++;
    return true;
  }

  uint32_t* arr = validation_buf_;

  copy_buf(arr, num_u32_);

  uint32_t inc_step = send_inc_step_;
  for (int idx = 0; idx < num_u32_; idx++) {
    if (arr[idx] != ref_arr[idx] + inc_step * iter_cnt_) {
      // TODO(chechenglin): add thread id, gpu idx, connection id to the log
      LOG(ERROR) << "Found mismatch during validation";
      if (idx < num_u32_ - 1) {
        std::stringstream ss;
        ss << "Peek forward: ";
        auto end_idx = std::min(num_u32_, (size_t)(idx + 16));
        for (auto k = idx; k < end_idx; k++) {
          ss << absl::StrFormat(" 0x%08x", arr[k]);
        }
        LOG(ERROR) << ss.str();

        ss.str("");
        ss << "Iter0 forward: ";
        for (auto k = idx; k < end_idx; k++) {
          ss << absl::StrFormat(" 0x%08x", ref_arr[k]);
        }
        LOG(ERROR) << ss.str();
      }
      if (0 < idx) {
        std::stringstream ss;
        ss << "Peek backward: ";
        auto start_idx = std::max(0, (int)(idx - 16));
        for (auto k = start_idx; k < idx; k++) {
          ss << absl::StrFormat(" 0x%08x", arr[k]);
        }
        LOG(ERROR) << ss.str();

        ss.str("");
        ss << "Iter0 backward: ";
        for (auto k = start_idx; k < idx; k++) {
          ss << absl::StrFormat(" 0x%08x", ref_arr[k]);
        }
        LOG(ERROR) << ss.str();
      }
      LOG(ERROR) << absl::StrFormat(
          "Error while validating at pos %d: 0x%08x vs 0x%08x", idx, arr[idx],
          ref_arr[idx] + inc_step * iter_cnt_);
      return false;
    }
  }
  iter_cnt_++;
  return true;
}
}  // namespace gpudirect_tcpxd
