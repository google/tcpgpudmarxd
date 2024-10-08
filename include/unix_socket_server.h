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

#include <absl/container/flat_hash_map.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
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

namespace gpudirect_tcpxd {

class UnixSocketServer {
  using ServiceFunc =
      std::function<void(UnixSocketMessage&&, UnixSocketMessage*, bool*)>;
  using AsyncServiceFunc = std::function<void(
      UnixSocketMessage&&, std::function<void(UnixSocketMessage&&, bool)>)>;
  using CleanupFunc = std::function<void(int)>;

 public:
  explicit UnixSocketServer(std::string path, ServiceFunc service_handler,
                            std::function<void()> service_setup = nullptr,
                            CleanupFunc cleanup_handler = nullptr);
  explicit UnixSocketServer(std::string path, AsyncServiceFunc service_handler,
                            std::function<void()> service_setup = nullptr,
                            CleanupFunc cleanup_handler = nullptr);
  ~UnixSocketServer();
  absl::Status Start();
  void Stop();

 private:
  int RegisterEvents(int fd, uint32_t events);
  int UnregisterFd(int fd);
  void EventLoop();
  void HandleListener(uint32_t events);
  void HandleClientCallback(int client, UnixSocketMessage&& response, bool fin);
  void HandleClient(int client_socket, uint32_t events);
  void RemoveClient(int client_socket);
  void AddConnectedClient(int socket);
  size_t NumConnections();
  std::pair<UnixSocketConnection*, bool> GetConnection(int client);

  void Worker();

  std::string path_;
  ServiceFunc service_handler_{nullptr};
  AsyncServiceFunc async_service_handler_{nullptr};
  std::function<void()> service_setup_{nullptr};
  CleanupFunc cleanup_handler_{nullptr};
  bool sync_handler_;

  struct sockaddr_un sockaddr_un_;
  size_t sockaddr_len_;
  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> event_thread_{nullptr};
  int listener_socket_{-1};
  int epoll_fd_{-1};

  absl::flat_hash_map<int, std::unique_ptr<UnixSocketConnection>>
      connected_clients_ ABSL_GUARDED_BY(&mu_);
  absl::flat_hash_map<int, bool> finished_
      ABSL_GUARDED_BY(&mu_);  // used by async handler to indicate this
                              // connection need to be closed
  absl::Mutex mu_;
};
}  // namespace gpudirect_tcpxd
#endif
