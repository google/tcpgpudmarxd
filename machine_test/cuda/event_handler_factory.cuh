#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_EVENT_HANDLER_FACTORY_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_EVENT_HANDLER_FACTORY_H_

#include <memory>
#include <string>

#include "machine_test/include/event_handler_interface.h"

namespace gpudirect_tcpxd {
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
  virtual bool UseCuda() = 0;
};

class TcpSendEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpSendEventHandlerFactory() override = default;
  bool UseCuda() override { return false; }
};

class TcpReceiveEventHandlerFactory : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpReceiveEventHandlerFactory() override = default;
  bool UseCuda() override { return false; }
};

class TcpSendTcpDirectEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpSendTcpDirectEventHandlerFactory() override = default;
  bool UseCuda() override { return false; }
};

class TcpReceiveTcpDirectEventHandlerFactory
    : public EventHandlerFactoryInterface {
 public:
  std::unique_ptr<EventHandlerInterface> New(std::string thread_id, int socket,
                                             size_t message_size,
                                             bool do_validation) override;
  ~TcpReceiveTcpDirectEventHandlerFactory() override = default;
  bool UseCuda() override { return false; }
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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }
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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }

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
  bool UseCuda() override { return true; }

 private:
  std::string gpu_pci_addr_;
};

std::unique_ptr<EventHandlerFactoryInterface> EventHandlerFactorySelector(
    std::string event_handler_name, std::string gpu_pci, std::string nic_pci);

}  // namespace gpudirect_tcpxd
#endif
