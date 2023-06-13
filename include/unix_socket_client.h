#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_CLIENT_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_CLIENT_H_

#include <memory>
#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/unix_socket_connection.h"
#include "experimental/users/chechenglin/tcpgpudmad/proto/unix_socket_message.proto.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"

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
