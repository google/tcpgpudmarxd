#include "machine_test/include/tcp_receive_event_handler.h"

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <sys/socket.h>

#include <memory>
#include <string>

namespace gpudirect_tcpxd {
TcpReceiveEventHandler::TcpReceiveEventHandler(std::string thread_id,
                                               int socket, size_t message_size,
                                               bool do_validation) {
  thread_id_ = thread_id + " [TCP-RECEIVE]";
  socket_ = socket;
  message_size_ = message_size;
  rx_buf_.reset(new char[message_size_]);
  rx_offset_ = 0;
  epoch_rx_bytes_ = 0;
  do_validation_ = do_validation;
  if (do_validation_) {
    validator_ = std::make_unique<ValidationReceiverCtx>(message_size);
  }
}

bool TcpReceiveEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLIN) {
    ssize_t ret = recv(socket_, &rx_buf_.get()[rx_offset_],
                       message_size_ - rx_offset_, MSG_DONTWAIT);
    PCHECK(ret >= 0 || (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)));
    if (ret >= 0) {
      epoch_rx_bytes_ += ret;
      rx_offset_ += ret;
      if (rx_offset_ >= message_size_) {
        rx_offset_ = 0;
        if (do_validation_) {
          validator_->ValidateRxData(
              /*init_buffer=*/
              [this](uint32_t* arr, int num_u32) {
                memcpy(arr, rx_buf_.get(), sizeof(uint32_t) * num_u32);
              },
              /*copy_buf=*/
              [this](uint32_t* arr, int num_u32) {
                memcpy(arr, rx_buf_.get(), sizeof(uint32_t) * num_u32);
              });
        }
      }
    }
  }
  return true;
}

double TcpReceiveEventHandler::GetRxBytes() {
  double ret = (double)epoch_rx_bytes_;
  epoch_rx_bytes_ = 0;
  return ret;
}
}  // namespace gpudirect_tcpxd
