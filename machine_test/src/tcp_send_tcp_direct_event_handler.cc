#include "machine_test/include/tcp_send_tcp_direct_event_handler.h"

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <linux/errqueue.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <memory>
#include <string>

#define MSG_SOCK_DEVMEM 0x2000000

namespace gpudirect_tcpxd {
TcpSendTcpDirectEventHandler::TcpSendTcpDirectEventHandler(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  thread_id_ = absl::StrFormat("%s [TCPSEND] socket [%d]", thread_id, socket);
  socket_ = socket;
  message_size_ = message_size;
  tx_buf_.reset(new char[message_size_]);
  tx_offset_ = 0;
  epoch_tx_bytes_ = 0;
  do_validation_ = do_validation;
  if (do_validation_) {
    validator_ = std::make_unique<ValidationSenderCtx>(message_size);
    validator_->InitSender([this](uint32_t* arr, int num_u32) {
      memcpy((void*)tx_buf_.get(), arr, sizeof(uint32_t) * num_u32);
    });
  }
}

bool TcpSendTcpDirectEventHandler::HandleEvents(unsigned events) {
  if (events & EPOLLOUT) {
    if (!HandleEPollOut()) return false;
  }
  if (events & EPOLLERR) {
    if (!HandleEPollErr()) return false;
  }
  return true;
}
bool TcpSendTcpDirectEventHandler::HandleEPollOut() {
  ssize_t ret =
      send(socket_, &(tx_buf_.get()[tx_offset_]), message_size_ - tx_offset_,
           MSG_DONTWAIT | MSG_ZEROCOPY | MSG_SOCK_DEVMEM);
  PCHECK(ret >= 0 || (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) ||
                      errno == ECONNRESET));
  if (errno == ECONNRESET) return false;
  if (ret >= 0) {
    epoch_tx_bytes_ += ret;
    tx_offset_ += ret;
    if (tx_offset_ >= message_size_) {
      tx_offset_ = 0;
      if (do_validation_) {
        validator_->UpdateSender([this](int num_u32, int send_inc_step) {
          uint32_t* arr = (uint32_t*)tx_buf_.get();
          for (int j = 0; j < num_u32; j++) {
            arr[j] += send_inc_step;
          }
        });
      }
    }
  }
  return true;
}

bool TcpSendTcpDirectEventHandler::HandleEPollErr() {
  char control[128];
  struct msghdr msg = {
      .msg_control = control,
      .msg_controllen = sizeof(control),
  };
  ssize_t ret = recvmsg(socket_, &msg, MSG_ERRQUEUE);
  if (ret == -1) {
    perror("error recv error msg");
    assert(false && "recvmsg");
  }
  struct sock_extended_err* serr;
  struct cmsghdr* cm;

  cm = CMSG_FIRSTHDR(&msg);
  while (cm) {
    if (cm->cmsg_level != SOL_IPV6 && cm->cmsg_type != IP_RECVERR)
      LOG(WARNING) << "Unexpected cmsg, level: " << cm->cmsg_level
                   << " type: " << cm->cmsg_type;

    serr = reinterpret_cast<struct sock_extended_err*>(CMSG_DATA(cm));
    if (serr->ee_errno != 0 || serr->ee_origin != SO_EE_ORIGIN_ZEROCOPY)
      LOG(WARNING) << "Unexpected serr, ee_errno: " << serr->ee_errno
                   << " ee_origin: " << serr->ee_origin;
    cm = CMSG_NXTHDR(&msg, cm);
  }
  return true;
}

double TcpSendTcpDirectEventHandler::GetTxBytes() {
  double ret = (double)epoch_tx_bytes_;
  epoch_tx_bytes_ = 0;
  return ret;
}
}  // namespace gpudirect_tcpxd
