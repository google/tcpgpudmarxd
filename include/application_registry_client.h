#ifndef NET_GPUDIRECTTCPX_RXBUFMGRCLIENT_APPLICATION_REGISTRY_CLIENT_H_
#define NET_GPUDIRECTTCPX_RXBUFMGRCLIENT_APPLICATION_REGISTRY_CLIENT_H_

#include <absl/status/status.h>

#include <memory>
#include <string>

#include "include/unix_socket_client.h"

namespace gpudirect_tcpxd {

class ApplicationRegistryClient {
 public:
  explicit ApplicationRegistryClient(std::string prefix);
  ~ApplicationRegistryClient();
  absl::Status Init();
  absl::Status Cleanup();

 private:
  std::string prefix_;
  std::unique_ptr<UnixSocketClient> client_;
};
}  // namespace gpudirect_tcpxd
#endif  // NET_GPUDIRECTTCPX_RXBUFMGRCLIENT_APPLICATION_REGISTRY_CLIENT_H_