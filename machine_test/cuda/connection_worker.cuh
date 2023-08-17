#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_CONNECTION_WORKER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_CONNECTION_WORKER_H_

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "include/flow_steer_ntuple.h"
#include "include/socket_helper.h"
#include "machine_test/cuda/event_handler_factory.cuh"
#include "machine_test/include/benchmark_common.h"

namespace gpudirect_tcpxd {

struct ConnectionWorkerConfig {
  bool do_validation;
  bool is_server;
  int num_sockets;
  size_t message_size;
  struct ThreadId thread_id;
  int server_port;
  union SocketAddress server_address;
  union SocketAddress client_address;
  std::string gpu_pci_addr;
};

class ConnectionWorker {
 public:
  explicit ConnectionWorker(
      struct ConnectionWorkerConfig worker_config,
      std::unique_ptr<EventHandlerFactoryInterface> ev_factory);
  ~ConnectionWorker();
  void Start(
      std::function<void(std::vector<std::string> errors)> post_run_callback);
  void Stop();
  double GetRxBytes();
  double GetTxBytes();
  int GpuIdx() { return gpu_idx_; }

 private:
  void Run();
  void ServerListen();
  void ClientConnect();

  std::unique_ptr<EventHandlerFactoryInterface> ev_factory_;
  std::vector<std::unique_ptr<EventHandlerInterface>> ev_handlers_;
  int gpu_idx_;
  std::string gpu_pci_addr_;
  bool do_validation_;
  size_t message_size_;
  int num_sockets_;
  union SocketAddress server_address_;
  bool is_server_;
  std::string thread_id_;
  union SocketAddress client_address_;
  FlowSteerNtuple fs_ntuple_;
  std::vector<int> connections_;
  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> thread_{nullptr};
  std::function<void(std::vector<std::string>)> post_run_callback_;
};

}  // namespace gpudirect_tcpxd
#endif