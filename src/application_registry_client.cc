#include "include/application_registry_client.h"

#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <unistd.h>

#include <string>

namespace gpudirect_tcpxd {
ApplicationRegistryClient::ApplicationRegistryClient(std::string prefix)
    : prefix_(prefix) {}

absl::Status ApplicationRegistryClient::Init() {
  const std::string server_addr = "rxdm_application_registry";

  client_ = std::make_unique<UnixSocketClient>(
      absl::StrFormat("%s/%s", prefix_, server_addr));

  UnixSocketMessage message;
  UnixSocketProto* proto = message.mutable_proto();
  ApplicationRegisterRequest* arr =
      proto->mutable_application_register_request();
  arr->set_register_client(true);

  UnixSocketMessage response;
  long pid = getpid();
  printf("%ld: connect and send\n", pid);
  if (auto status = ConnectAndSendMessage(message, &response, client_.get());
      !status.ok()) {
    return absl::InternalError(
        absl::StrFormat("register pid %ld failed connect and send: %s", pid,
                        status.ToString()));
  }

  return absl::OkStatus();
}

ApplicationRegistryClient::~ApplicationRegistryClient() { (void)Cleanup(); }

absl::Status ApplicationRegistryClient::Cleanup() {
  const std::string server_addr = "rxdm_application_registry";

  UnixSocketMessage message;
  UnixSocketProto* proto = message.mutable_proto();
  ApplicationRegisterRequest* arr =
      proto->mutable_application_register_request();
  arr->set_register_client(false);

  UnixSocketMessage response;
  if (auto status = ConnectAndSendMessage(message, &response, client_.get());
      !status.ok()) {
    long pid = getpid();
    return absl::InternalError(
        absl::StrFormat("deregister pid %ld failed connect and send: %s", pid,
                        status.ToString()));
  }

  return absl::OkStatus();
}
}  // namespace gpudirect_tcpxd