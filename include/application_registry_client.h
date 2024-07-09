/*
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

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