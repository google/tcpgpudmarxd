#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_EVENT_HANDLER_FACTORY_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_EVENT_HANDLER_FACTORY_H_

#include <memory>
#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/benchmark/event_handler_interface.h"

namespace tcpdirect {
enum TrafficDirection {
  TCP_SENDER,
  TCP_RECEIVER,
};

class EventHandlerFactoryInterface {
 public:
  virtual std::unique_ptr<EventHandlerInterface> New(std::string thread_id,
                                                     int socket,
                                                     size_t message_size,
                                                     bool do_validation) = 0;
  virtual ~EventHandlerFactoryInterface() {}
};

class TcpSendEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpSendEventHandlerFactory() override = default;
};

class TcpReceiveEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpReceiveEventHandlerFactory() override = default;
};

class TcpSendTcpDirectEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpSendTcpDirectEventHandlerFactory() override = default;
};

class TcpReceiveTcpDirectEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpReceiveTcpDirectEventHandlerFactory() override = default;
};

class DmabufSendEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  DmabufSendEventHandlerFactory(std::string gpu_pci_addr,
                                std::string nic_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr), nic_pci_addr_(nic_pci_addr) {}
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~DmabufSendEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
  std::string nic_pci_addr_;
};

class GpuReceiveEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  GpuReceiveEventHandlerFactory(std::string gpu_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr) {}
  virtual std::unique_ptr<EventHandlerInterface> New(
      std::string thread_id, int socket, size_t message_size,
      bool do_validation) override;
  ~GpuReceiveEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
};

// Dummy classes for testing

class DummyPageGpuSendEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  DummyPageGpuSendEventHandlerFactory() {}
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~DummyPageGpuSendEventHandlerFactory() = default;
};

class GpuSendMissFlagEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  GpuSendMissFlagEventHandlerFactory(std::string gpu_pci_addr,
                                     std::string nic_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr), nic_pci_addr_(nic_pci_addr) {}
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~GpuSendMissFlagEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
  std::string nic_pci_addr_;
};

class GpuSendMixTcpEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  GpuSendMixTcpEventHandlerFactory(std::string gpu_pci_addr,
                                   std::string nic_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr), nic_pci_addr_(nic_pci_addr) {}
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~GpuSendMixTcpEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
  std::string nic_pci_addr_;
};

class GpuSendOobEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  GpuSendOobEventHandlerFactory(std::string gpu_pci_addr,
                                std::string nic_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr), nic_pci_addr_(nic_pci_addr) {}
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~GpuSendOobEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
  std::string nic_pci_addr_;
};

class GpuReceiveMissFlagEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  GpuReceiveMissFlagEventHandlerFactory(std::string gpu_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr) {}
  virtual std::unique_ptr<EventHandlerInterface> New(
      std::string thread_id, int socket, size_t message_size,
      bool do_validation) override;
  ~GpuReceiveMissFlagEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
};

class GpuReceiveMixTcpEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  GpuReceiveMixTcpEventHandlerFactory(std::string gpu_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr) {}
  virtual std::unique_ptr<EventHandlerInterface> New(
      std::string thread_id, int socket, size_t message_size,
      bool do_validation) override;
  ~GpuReceiveMixTcpEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
};

class GpuReceiveTokenFreeEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  GpuReceiveTokenFreeEventHandlerFactory(std::string gpu_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr) {}
  virtual std::unique_ptr<EventHandlerInterface> New(
      std::string thread_id, int socket, size_t message_size,
      bool do_validation) override;
  ~GpuReceiveTokenFreeEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
};

class GpuReceiveNoTokenFreeEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  GpuReceiveNoTokenFreeEventHandlerFactory(std::string gpu_pci_addr)
      : gpu_pci_addr_(gpu_pci_addr) {}
  virtual std::unique_ptr<EventHandlerInterface> New(
      std::string thread_id, int socket, size_t message_size,
      bool do_validation) override;
  ~GpuReceiveNoTokenFreeEventHandlerFactory() = default;

 private:
  std::string gpu_pci_addr_;
};

std::unique_ptr<EventHandlerFactoryInterface> EventHandlerFactorySelector(
    std::string event_handler_name, std::string gpu_pci, std::string nic_pci);

}  // namespace tcpdirect
#endif
