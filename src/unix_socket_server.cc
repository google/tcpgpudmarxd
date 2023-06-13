#include "experimental/users/chechenglin/tcpgpudmad/include/unix_socket_server.h"

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

#include "experimental/users/chechenglin/tcpgpudmad/include/unix_socket_connection.h"
#include "experimental/users/chechenglin/tcpgpudmad/proto/unix_socket_message.proto.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_format.h"

namespace tcpdirect {

namespace {
constexpr int kDefaultBacklog{1024};
}  // namespace

UnixSocketServer::UnixSocketServer(std::string path,
                                   ServiceFunc service_handler,
                                   std::function<void()> service_setup)
    : path_(path),
      service_handler_(std::move(service_handler)),
      service_setup_(std::move(service_setup)) {
  sockaddr_un_.sun_family = AF_UNIX;
  strcpy(sockaddr_un_.sun_path, path_.c_str());
  sockaddr_len_ =
      strlen(sockaddr_un_.sun_path) + sizeof(sockaddr_un_.sun_family);
}

UnixSocketServer::~UnixSocketServer() { Stop(); }

absl::Status UnixSocketServer::Start() {
  if (path_.empty())
    return absl::InvalidArgumentError("Missing file path to domain socket.");

  if (service_handler_ == nullptr)
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
    std::vector<struct epoll_event> events(connected_clients_.size() + 1);
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
    unsigned int peer_addr_len;
    int socket = accept4(listener_socket_, (struct sockaddr*)&peer_addr,
                         &peer_addr_len, 0);
    connected_clients_[socket] = std::make_unique<UnixSocketConnection>(socket);
    RegisterEvents(socket, EPOLLIN | EPOLLOUT);
  }
}

void UnixSocketServer::HandleClient(int client, uint32_t events) {
  UnixSocketConnection& connection = *connected_clients_[client];
  bool fin{false};
  if (events & EPOLLIN) {
    if (connection.Receive()) {
      if (connection.HasNewMessageToRead()) {
        UnixSocketMessage response;
        service_handler_(connection.ReadMessage(), &response, &fin);
        connection.AddMessageToSend(std::move(response));
      }
    } else {
      fin = true;
    }
  }
  if (events & EPOLLOUT) {
    if (!connection.Send()) {
      fin = true;
    }
  }
  if ((events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) || fin) {
    RemoveClient(client);
    return;
  }
}

void UnixSocketServer::RemoveClient(int client_socket) {
  UnregisterFd(client_socket);
  connected_clients_.erase(client_socket);
}
}  // namespace tcpdirect
