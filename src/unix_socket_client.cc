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

#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <memory>
#include <utility>

#include "include/unix_socket_connection.h"
#include "proto/unix_socket_message.pb.h"
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>

namespace tcpdirect {
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

}  // namespace tcpdirect
