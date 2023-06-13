#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_CLIENT_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_CLIENT_H_

#include <memory>
#include <string>

#include "include/unix_socket_connection.h"
#include "proto/unix_socket_message.proto.h"
#include <absl/status/status.h>
#include <absl/status/statusor.h>

namespace tcpdirect {
class UnixSocketClient {
 public:
  explicit UnixSocketClient(std::string path) : path_(path) {}
  absl::Status Connect();
  absl::StatusOr<UnixSocketMessage> Receive();
  void Send(UnixSocketMessage msg);

 private:
  std::unique_ptr<UnixSocketConnection> conn_;
  std::string path_;
};
}  // namespace tcpdirect
#endif
