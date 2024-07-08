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

#include "include/unix_socket_server.h"

#include <absl/functional/bind_front.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>
#include <errno.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "include/unix_socket_connection.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {

namespace {
constexpr int kDefaultBacklog{1024};
}  // namespace

UnixSocketServer::UnixSocketServer(std::string path,
                                   ServiceFunc service_handler,
                                   std::function<void()> service_setup,
                                   CleanupFunc cleanup_handler)
    : path_(path),
      service_handler_(std::move(service_handler)),
      service_setup_(std::move(service_setup)),
      cleanup_handler_(std::move(cleanup_handler)),
      sync_handler_(true) {
  sockaddr_un_.sun_family = AF_UNIX;
  strcpy(sockaddr_un_.sun_path, path_.c_str());
  sockaddr_len_ =
      strlen(sockaddr_un_.sun_path) + sizeof(sockaddr_un_.sun_family);
}

UnixSocketServer::UnixSocketServer(std::string path,
                                   AsyncServiceFunc service_handler,
                                   std::function<void()> service_setup,
                                   CleanupFunc cleanup_handler)
    : path_(path),
      async_service_handler_(std::move(service_handler)),
      service_setup_(std::move(service_setup)),
      cleanup_handler_(std::move(cleanup_handler)),
      sync_handler_(false) {
  sockaddr_un_.sun_family = AF_UNIX;
  strcpy(sockaddr_un_.sun_path, path_.c_str());
  sockaddr_len_ =
      strlen(sockaddr_un_.sun_path) + sizeof(sockaddr_un_.sun_family);
}

UnixSocketServer::~UnixSocketServer() { Stop(); }

absl::Status UnixSocketServer::Start() {
  if (path_.empty())
    return absl::InvalidArgumentError("Missing file path to domain socket.");

  if (service_handler_ == nullptr && async_service_handler_ == nullptr)
    return absl::InvalidArgumentError("Missing service handler.");

  if (unlink(path_.c_str())) {
    if (errno != ENOENT) {
      return absl::ErrnoToStatus(errno,
                                 absl::StrFormat("unlink() error: %d", errno));
    }
  }

  listener_socket_ = socket(AF_UNIX, SOCK_STREAM, 0);

  if (listener_socket_ < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("socket() error: %d", error_number));
  }

  if (bind(listener_socket_, (struct sockaddr*)&sockaddr_un_, sockaddr_len_) !=
      0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("bind() error: %d", error_number));
  }

  if (listen(listener_socket_, kDefaultBacklog) != 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("listen() error: %d", error_number));
  }

  running_.store(true);

  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("epoll_create1() error: %d", error_number));
  }

  if (RegisterEvents(listener_socket_, EPOLLIN) < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("epoll_ctl() error: %d", error_number));
  }

  event_thread_ = std::make_unique<std::thread>([this] { EventLoop(); });

  return absl::OkStatus();
}

void UnixSocketServer::Stop() {
  running_.store(false);

  if (event_thread_ && event_thread_->joinable()) {
    event_thread_->join();
  }
  // Remove all connected clients, closing all existing UDS connections
  connected_clients_.clear();
  if (listener_socket_ >= 0) {
    close(listener_socket_);
  }
}

int UnixSocketServer::RegisterEvents(int fd, uint32_t events) {
  struct epoll_event event;
  event.events = events;
  event.data.fd = fd;
  return epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &event);
}

int UnixSocketServer::UnregisterFd(int fd) {
  return epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
}

void UnixSocketServer::EventLoop() {
  // Setting up the context for this serving thread.
  if (service_setup_) {
    service_setup_();
  }

  while (running_) {
    std::vector<struct epoll_event> events(NumConnections() + 1);
    int nevents = epoll_wait(epoll_fd_, events.data(), events.size(),
                             std::chrono::milliseconds(100).count());
    for (int i = 0; i < nevents; ++i) {
      const struct epoll_event& event = events[i];
      if (event.data.fd == listener_socket_) {
        HandleListener(event.events);
      } else {
        HandleClient(event.data.fd, event.events);
      }
    }
  }
}

void UnixSocketServer::HandleListener(uint32_t events) {
  if (events & EPOLLERR) {
    int error_number = errno;
    std::cerr << absl::StrFormat("Listener socket error, errno: %d",
                                 error_number);
    running_.store(false);
  }
  if (events & EPOLLIN) {
    struct sockaddr_un peer_addr;
    unsigned int peer_addr_len = sizeof(peer_addr);
    int socket = accept4(listener_socket_, (struct sockaddr*)&peer_addr,
                         &peer_addr_len, 0);
    if (socket < 0) {
      PLOG(ERROR) << absl::StrFormat("accept4 error: %d, errno: ", socket);
    } else {
      AddConnectedClient(socket);
      RegisterEvents(socket, EPOLLIN | EPOLLOUT);
    }
  }
}

void UnixSocketServer::HandleClientCallback(int client,
                                            UnixSocketMessage&& response,
                                            bool fin) {
  // Need to grab the lock for the entire function to make sure the connection
  // is not cleaned up during this function
  absl::MutexLock lock(&mu_);
  auto it = connected_clients_.find(client);
  if (it == connected_clients_.end()) {
    return;
  }
  it->second->AddMessageToSend(std::move(response));
  finished_[client] = true;
}

void UnixSocketServer::HandleClient(int client, uint32_t events) {
  auto [connection, fin] = GetConnection(client);
  if (!connection) {
    return;
  }

  if (events & EPOLLIN) {
    if (connection->Receive()) {
      if (connection->HasNewMessageToRead()) {
        if (!sync_handler_) {
          async_service_handler_(
              connection->ReadMessage(),
              absl::bind_front(&UnixSocketServer::HandleClientCallback, this,
                               client));
        } else {
          UnixSocketMessage response;
          service_handler_(connection->ReadMessage(), &response, &fin);
          connection->AddMessageToSend(std::move(response));
        }
      }
    } else {
      if (errno) {
        PLOG(ERROR) << absl::StrFormat("Receive error on client %d, errno: ",
                                       client);
      }
      fin = true;
    }
  }
  if (events & EPOLLOUT) {
    if (!connection->Send()) {
      if (errno) {
        PLOG(ERROR) << absl::StrFormat("Send failure on client %d, errno: ",
                                       client);
      }
      fin = true;
    }
  }
  if ((events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) || fin) {
    if (events & EPOLLERR) {
      LOG(ERROR) << absl::StrFormat("EPOLLERR on client %d", client);
    }
    if(cleanup_handler_) {
      cleanup_handler_(client);
    }
    RemoveClient(client);
    return;
  }
}

void UnixSocketServer::RemoveClient(int client_socket) {
  UnregisterFd(client_socket);
  absl::MutexLock lock(&mu_);
  connected_clients_.erase(client_socket);
  finished_.erase(client_socket);
}

void UnixSocketServer::AddConnectedClient(int socket) {
  absl::MutexLock lock(&mu_);
  connected_clients_[socket] = std::make_unique<UnixSocketConnection>(socket);
  finished_[socket] = false;
}

size_t UnixSocketServer::NumConnections() {
  absl::MutexLock lock(&mu_);
  return connected_clients_.size();
}

std::pair<UnixSocketConnection*, bool> UnixSocketServer::GetConnection(
    int client) {
  absl::MutexLock lock(&mu_);
  auto it = connected_clients_.find(client);
  if (it == connected_clients_.end()) {
    LOG(ERROR) << "Connection of " << client << " is not found";
    return {nullptr, true};
  }
  return {it->second.get(), finished_[client]};
}

}  // namespace gpudirect_tcpxd
