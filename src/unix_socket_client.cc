// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/unix_socket_client.h"

#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <memory>
#include <thread>
#include <utility>

#include "include/unix_socket_connection.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {
UnixSocketClient::UnixSocketClient(std::string path,
                                   std::function<int()> vf_reset_cb)
    : path_(path), vf_reset_cb_(vf_reset_cb) {
  // We only want an epoll instance if vf_reset_cb.
  // We'll only poll for EPOLLHUP to trigger the callback function.
  if (vf_reset_cb_) {
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
      int err_num = errno;
      PLOG(ERROR) << absl::StrFormat("epoll_create1() error: %d", err_num);
    }
  }
}

UnixSocketClient::~UnixSocketClient() {
  if (epoll_fd_ > 0) {
    close(epoll_fd_);
    epoll_fd_ = -1;
    epoll_thread_.join();
  }
}

absl::Status UnixSocketClient::Connect() {
  if (path_.empty())
    return absl::InvalidArgumentError("Missing file path to domain socket.");

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);

  if (fd < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("socket() error: %d", error_number));
  }

  struct sockaddr_un server_addr;
  int server_addr_len;
  server_addr.sun_family = AF_UNIX;
  strcpy(server_addr.sun_path, path_.c_str());
  server_addr_len =
      strlen(server_addr.sun_path) + sizeof(server_addr.sun_family);
  if (connect(fd, (struct sockaddr*)&server_addr, server_addr_len) < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("connect() error: %d", error_number));
  }

  // epoll_fd_ exists only if creator of this UnixSocketClient passed a
  // callback function in case of VF reset.
  if (epoll_fd_ > 0) {
    struct epoll_event event = {.events = EPOLLRDHUP};
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &event);

    // Start a thread that epolls for socket hang-up.
    // When a hang-up occurs, then we trigger the VF reset callback function.
    epoll_thread_ = std::thread([this]() {
      struct epoll_event event;
      int ret;
      while (epoll_fd_ > 0) {
        ret = epoll_wait(epoll_fd_, &event, 1,
                         std::chrono::milliseconds(100).count());

        if (ret < 0) {
          PLOG(ERROR) << "epoll_wait error: ";
          return;
        }
        if (ret == 1) {
          vf_reset_cb_();
          return;
        }
      }
    });
  }

  conn_ = std::make_unique<UnixSocketConnection>(fd);
  return absl::OkStatus();
}

absl::StatusOr<UnixSocketMessage> UnixSocketClient::Receive() {
  while (!conn_->HasNewMessageToRead()) {
    if (!conn_->Receive()) {
      int error_number = errno;
      return absl::ErrnoToStatus(
          errno, absl::StrFormat("receive() error: %d", error_number));
    }
  }
  return conn_->ReadMessage();
}

void UnixSocketClient::Send(UnixSocketMessage msg) {
  conn_->AddMessageToSend(std::move(msg));
  while (conn_->HasPendingMessageToSend()) {
    if (!conn_->Send()) {
      break;
    }
  }
}

}  // namespace gpudirect_tcpxd
