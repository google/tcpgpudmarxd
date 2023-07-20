#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_VALIDATION_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_VALIDATION_H_

#include <stdint.h>

#include <functional>

//// ValidationSenderCtx /////

namespace gpudirect_tcpxd {
class ValidationSenderCtx {
 public:
  explicit ValidationSenderCtx(size_t message_size);
  ~ValidationSenderCtx();
  void InitSender(const std::function<void(uint32_t*, int)>& commit_buffer);
  void UpdateSender(const std::function<void(int, int)>& update_buffer);

 private:
  uint32_t send_inc_step_;
  size_t num_u32_;
  int iter_cnt_;
  uint32_t* iter0_data_{nullptr};
};

//// ValidationReceiverCtx /////

class ValidationReceiverCtx {
 public:
  explicit ValidationReceiverCtx(size_t message_size);
  virtual ~ValidationReceiverCtx();
  void InitRxBuf(const std::function<void(uint32_t*)>& init_buffer);
  void RxBufToValidationBuf(const std::function<void(uint32_t*)>& copy_buf);
  bool ValidateRxData(const std::function<void(uint32_t*, int)>& init_buffer,
                      const std::function<void(uint32_t*, int)>& copy_buf);

 private:
  uint32_t* iter0_data_{nullptr};
  uint32_t* validation_buf_{nullptr};
  uint32_t send_inc_step_;
  size_t num_u32_;
  int iter_cnt_;
};

}  // namespace gpudirect_tcpxd
#endif
