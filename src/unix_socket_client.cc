#include "experimental/users/chechenglin/tcpgpudmad/include/unix_socket_client.h"

#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <memory>
#include <utility>

#include "experimental/users/chechenglin/tcpgpudmad/include/unix_socket_connection.h"
#include "experimental/users/chechenglin/tcpgpudmad/proto/unix_socket_message.proto.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_format.h"

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
