#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <linux/ethtool.h>
#include <sys/epoll.h>
#include <sys/socket.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "cuda/common.cuh"
#include "include/rx_rule_client.h"
#include "include/socket_helper.h"
#include "machine_test/cuda/connection_worker.cuh"

namespace gpudirect_tcpxd {

ConnectionWorker::ConnectionWorker(
    struct ConnectionWorkerConfig worker_config,
    std::unique_ptr<EventHandlerFactoryInterface> ev_factory)
    : ev_factory_(std::move(ev_factory)) {
  gpu_idx_ = worker_config.thread_id.gpu_idx;
  gpu_pci_addr_ = worker_config.gpu_pci_addr;
  do_validation_ = worker_config.do_validation;
  message_size_ = worker_config.message_size;
  num_sockets_ = worker_config.num_sockets;
  server_address_ = worker_config.server_address;
  SetAddressPort(&server_address_, worker_config.server_port);
  client_address_ = worker_config.client_address;
  SetAddressPort(&client_address_, worker_config.server_port);

  union SocketAddress *from{nullptr};
  union SocketAddress *to{nullptr};

  if (worker_config.is_server) {
    is_server_ = true;
    thread_id_ = absl::StrFormat("Server - GPU [%d] Thread [%d]",
                                 worker_config.thread_id.gpu_idx,
                                 worker_config.thread_id.per_gpu_thread_idx);
    from = &client_address_;
    to = &server_address_;
  } else {
    is_server_ = false;
    thread_id_ = absl::StrFormat("Client - GPU [%d] Thread [%d]",
                                 worker_config.thread_id.gpu_idx,
                                 worker_config.thread_id.per_gpu_thread_idx);
    from = &server_address_;
    to = &client_address_;
  }
  if (server_address_.sa.sa_family == AF_INET) {
    fs_ntuple_.flow_type = TCP_V4_FLOW;
    memcpy(&fs_ntuple_.src_sin, &from->sin, sizeof(fs_ntuple_.src_sin));
    memcpy(&fs_ntuple_.dst_sin, &to->sin, sizeof(fs_ntuple_.dst_sin));
  } else {
    fs_ntuple_.flow_type = TCP_V6_FLOW;
    memcpy(&fs_ntuple_.src_sin6, &from->sin6, sizeof(fs_ntuple_.src_sin6));
    memcpy(&fs_ntuple_.dst_sin6, &to->sin6, sizeof(fs_ntuple_.dst_sin6));
  }
}

void ConnectionWorker::Stop() {
  LOG(INFO) << thread_id_ << ": stopping...";
  running_.store(false, std::memory_order_release);
  if (thread_ && thread_->joinable()) {
    thread_->join();
    thread_.reset(nullptr);
  }
}

ConnectionWorker::~ConnectionWorker() {
  Stop();
  for (int fd : connections_) {
    close(fd);
  }
  for (auto &ev : ev_handlers_) {
    ev.reset();
  }
}

void ConnectionWorker::Start(
    std::function<void(std::vector<std::string>)> post_run_callback) {
  post_run_callback_ = post_run_callback;
  thread_ = std::make_unique<std::thread>([this] { Run(); });
}

void ConnectionWorker::ServerListen() {
  int listen_fd = CreateTcpSocket(server_address_.sa.sa_family);
  SetReuseAddr(listen_fd);
  EnableTcpZeroCopy(listen_fd);
  BindAndListen(listen_fd, &server_address_, std::max(num_sockets_, 100));
  for (int i = 0; i < num_sockets_; i++) {
    union SocketAddress client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    int fd = accept(listen_fd, &client_addr.sa, &client_addr_len);
    connections_.emplace_back(fd);
    LOG(INFO) << thread_id_ << ": got connection from "
              << AddressToStr(&client_addr);
    int opt = 0;
    socklen_t optlen = sizeof(opt);
    if (getsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &opt, &optlen)) {
      perror("getsockopt(): ");
      exit(EXIT_FAILURE);
    }
    CHECK(opt == 1);
  }
  close(listen_fd);
}

void ConnectionWorker::ClientConnect() {
  socklen_t addrlen = (client_address_.sa.sa_family == AF_INET)
                          ? sizeof(struct sockaddr_in)
                          : sizeof(struct sockaddr_in6);
  SetAddressPort(&client_address_, 0);
  for (int i = 0; i < num_sockets_; i++) {
    int fd = CreateTcpSocket(server_address_.sa.sa_family);
    PCHECK(bind(fd, &client_address_.sa, addrlen) == 0);
    EnableTcpZeroCopy(fd);
    ConnectWithRetry(fd, &server_address_);
    connections_.emplace_back(fd);
    LOG(INFO) << thread_id_ << ": established " << i + 1 << " connections with "
              << AddressToStr(&server_address_) << " via "
              << AddressToStr(&client_address_);
  }
}

double ConnectionWorker::GetRxBytes() {
  double rx_bytes = 0.0;
  for (auto &ev_hdlr : ev_handlers_) {
    rx_bytes += ev_hdlr->GetRxBytes();
  }
  return rx_bytes;
}

double ConnectionWorker::GetTxBytes() {
  double tx_bytes = 0.0;
  for (auto &ev_hdlr : ev_handlers_) {
    tx_bytes += ev_hdlr->GetTxBytes();
  }
  return tx_bytes;
}

void ConnectionWorker::Run() {
  if (ev_factory_->UseCuda()) {
    // Setup Cuda Context
    CUDA_ASSERT_SUCCESS(cudaSetDevice(gpu_idx_));
    // Add Flow Steering Rule
    RxRuleClient client{"/tmp"};
    if (auto status = client.UpdateFlowSteerRule(FlowSteerRuleOp::CREATE,
                                                 fs_ntuple_, gpu_pci_addr_);
        !status.ok()) {
      LOG(ERROR) << thread_id_ << ": Failed to create flow steering rule. ";
    }
  }

  // Setup connections
  LOG(INFO) << thread_id_ << ": starting connections: ";

  if (is_server_) {
    ServerListen();
  } else {
    ClientConnect();
  }

  // Create event handlers
  for (auto conn : connections_) {
    ev_handlers_.emplace_back(
        ev_factory_->New(thread_id_, conn, message_size_, do_validation_));
  }

  // Create epoll
  int epoll_fd = epoll_create1(0);
  for (int i = 0; i < connections_.size(); i++) {
    struct epoll_event event;
    event.events = ev_handlers_[i]->InterestedEvents();
    event.data.u32 = i;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, connections_[i], &event);
  }

  // Dispatch epoll events
  std::vector<struct epoll_event> events(connections_.size());
  running_.store(true, std::memory_order_release);
  while (running_.load(std::memory_order_relaxed)) {
    int nevent = epoll_wait(epoll_fd, events.data(), events.size(),
                            std::chrono::milliseconds(500).count());
    for (int idx = 0; idx < nevent; idx++) {
      const struct epoll_event &event = events[idx];
      int i = event.data.u32;
      if (!ev_handlers_[i]->HandleEvents(event.events)) {
        running_.store(false, std::memory_order_release);
        break;
      }
    }
  }
  LOG(ERROR) << thread_id_ << ": finishes run.";
  std::vector<std::string> errors;
  for (auto &ev_hdlr : ev_handlers_) {
    errors.push_back(ev_hdlr->Error());
  }
  if (ev_factory_->UseCuda()) {
    RxRuleClient client{"/tmp"};
    if (auto status = client.UpdateFlowSteerRule(FlowSteerRuleOp::DELETE,
                                                 fs_ntuple_, gpu_pci_addr_);
        !status.ok()) {
      LOG(ERROR) << thread_id_ << ": Failed to delete flow steering rule. ";
    }
  }
  post_run_callback_(errors);
}
}  // namespace gpudirect_tcpxd
