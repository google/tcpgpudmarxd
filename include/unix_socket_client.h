/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_CLIENT_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_CLIENT_H_

#include <absl/status/status.h>
#include <absl/status/statusor.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

#include "include/unix_socket_connection.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {
class UnixSocketClient {
 public:
  explicit UnixSocketClient(std::string path,
                            std::function<int()> vf_reset_cb = nullptr);
  ~UnixSocketClient();
  absl::Status Connect();
  absl::StatusOr<UnixSocketMessage> Receive();
  void Send(UnixSocketMessage msg);
  bool IsConnected (){
    return conn_ != NULL;
  }

 private:
  std::unique_ptr<UnixSocketConnection> conn_;
  std::string path_;
  std::atomic_int epoll_fd_{-1};
  std::function<int()>
      vf_reset_cb_; /* callback function for when a VF reset occurs */
  std::thread epoll_thread_;
};

absl::Status ConnectAndSendMessage(UnixSocketMessage message,
                                   UnixSocketMessage* response,
                                   UnixSocketClient* client);

}  // namespace gpudirect_tcpxd
#endif
