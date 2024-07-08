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

#include "include/application_registry_manager.h"

#include <absl/functional/bind_front.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>

#include <memory>
#include <string>

namespace gpudirect_tcpxd {
namespace {
bool InvalidApplicationRegisterRequest(const UnixSocketMessage& request) {
  return !request.has_proto() ||
         !(request.proto().has_application_register_request());
}

UnixSocketMessage ConstructErrorResponse(const absl::Status& status) {
  UnixSocketMessage response;
  UnixSocketProto* proto = response.mutable_proto();
  std::string* buffer = proto->mutable_raw_bytes();
  proto->mutable_status()->set_code(status.raw_code());
  proto->mutable_status()->set_message(status.ToString());
  buffer->append(absl::StrFormat("Failed to process request, error: %s.",
                                 status.ToString()));
  return response;
}

}  // namespace

ApplicationRegistryManager::ApplicationRegistryManager(
    const std::string& prefix, pthread_t main_id)
    : prefix_(prefix), main_id_(main_id), connection_counts_(0) {}

absl::Status ApplicationRegistryManager::Init() {
  return AddApplicationRegistryServer();
}

void ApplicationRegistryManager::Cleanup() { server_->Stop(); }

absl::Status ApplicationRegistryManager::AddApplicationRegistryServer() {
  const std::string suffix = "rxdm_application_registry";
  std::string server_addr = absl::StrFormat("%s/%s", prefix_, suffix);
  LOG(INFO) << absl::StrFormat("Starting ApplicationRegistry server at %s",
                               server_addr);
  server_ = std::make_unique<UnixSocketServer>(
      server_addr,
      absl::bind_front(&ApplicationRegistryManager::ApplicationRegistryHandler,
                       this),
      nullptr,
      absl::bind_front(&ApplicationRegistryManager::HandleClientDrop, this));
  return server_->Start();
}

void ApplicationRegistryManager::ApplicationRegistryHandler(
    UnixSocketMessage&& request, UnixSocketMessage* response, bool* fin) {
  if (InvalidApplicationRegisterRequest(request)) {
    *response = ConstructErrorResponse(
        absl::InvalidArgumentError("Invalid Application Register Request"));
    *fin = true;
    return;
  }

  // Process request, if fail, will do graceful shutdown
  if (request.proto().application_register_request().has_register_client() &&
      !request.proto().application_register_request().register_client()) {
    *fin = true;
  } else {
    absl::MutexLock lock(&mu_);
    connection_counts_ += 1;
  }

  UnixSocketProto* proto = response->mutable_proto();
  std::string* buffer = proto->mutable_raw_bytes();
  buffer->append("Ok.");
}

void ApplicationRegistryManager::HandleClientDrop(int client_socket) {
  bool need_exit = false;
  {
    absl::MutexLock lock(&mu_);
    connection_counts_ -= 1;
    if (connection_counts_ <= 0) {
      need_exit = true;
    }
  }
  if (need_exit) {
    pthread_kill(main_id_, SIGTERM);
  }
}

}  // namespace gpudirect_tcpxd