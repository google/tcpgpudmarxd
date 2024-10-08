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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_APPLICATION_REGISTRY_MANAGER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_APPLICATION_REGISTRY_MANAGER_H_
#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>
#include <absl/status/status.h>
#include <net/if.h>
#include <pthread.h>
#include <memory>
#include <string>
#include "include/unix_socket_server.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {
class ApplicationRegistryManager {
 public:
  explicit ApplicationRegistryManager(const std::string& prefix, pthread_t main_id);
  ~ApplicationRegistryManager() { Cleanup(); }
  absl::Status Init();
  void Cleanup();
 private:
  void ApplicationRegistryHandler(UnixSocketMessage&& request,
                                  UnixSocketMessage* response, bool* fin);
  void HandleClientDrop(int client_socket);

  absl::Status AddApplicationRegistryServer();
  std::string prefix_;
  pthread_t main_id_;

  absl::Mutex mu_;
  int connection_counts_;
  std::unique_ptr<UnixSocketServer> server_;
};
}  // namespace gpudirect_tcpxd
#endif