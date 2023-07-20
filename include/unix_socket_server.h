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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_SERVER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_UNIX_SOCKET_SERVER_H_

#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/un.h>

#include <atomic>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "include/unix_socket_connection.h"
#include "proto/unix_socket_message.pb.h"
#include <absl/container/flat_hash_map.h>
#include <absl/status/status.h>

namespace gpudirect_tcpxd {

class UnixSocketServer {
  using ServiceFunc =
      std::function<void(UnixSocketMessage &&, UnixSocketMessage *, bool *)>;

 public:
  explicit UnixSocketServer(std::string path, ServiceFunc service_handler,
                            std::function<void()> service_setup = nullptr);
  ~UnixSocketServer();
  absl::Status Start();
  void Stop();

 private:
  int RegisterEvents(int fd, uint32_t events);
  int UnregisterFd(int fd);
  void EventLoop();
  void HandleListener(uint32_t events);
  void HandleClient(int client_socket, uint32_t events);
  void RemoveClient(int client_socket);

  void Worker();

  std::string path_;
  ServiceFunc service_handler_{nullptr};
  std::function<void()> service_setup_{nullptr};

  struct sockaddr_un sockaddr_un_;
  size_t sockaddr_len_;
  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> event_thread_{nullptr};
  int listener_socket_{-1};
  int epoll_fd_{-1};

  absl::flat_hash_map<int, std::unique_ptr<UnixSocketConnection>>
      connected_clients_;
};
}  // namespace gpudirect_tcpxd
#endif
