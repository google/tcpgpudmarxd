#include <absl/log/log.h>

#include <cstdlib>

#include "machine_test/cuda/cuda_dmabuf_gpu_page_allocator.cuh"
#include "machine_test/cuda/event_handler_factory.cuh"
#include "machine_test/cuda/gpu_receive_event_handler.cuh"
#include "machine_test/cuda/gpu_receive_miss_flag_event_handler.cuh"
#include "machine_test/cuda/gpu_receive_mix_tcp_event_handler.cuh"
#include "machine_test/cuda/gpu_receive_no_token_free_event_handler.cuh"
#include "machine_test/cuda/gpu_receive_token_free_event_handler.cuh"
#include "machine_test/cuda/gpu_send_event_handler.cuh"
#include "machine_test/cuda/gpu_send_event_handler_miss_flag.cuh"
#include "machine_test/cuda/gpu_send_event_handler_mix_tcp.cuh"
#include "machine_test/cuda/gpu_send_oob_event_handler.cuh"
#include "machine_test/include/tcp_receive_event_handler.h"
#include "machine_test/include/tcp_receive_tcp_direct_event_handler.h"
#include "machine_test/include/tcp_send_event_handler.h"
#include "machine_test/include/tcp_send_tcp_direct_event_handler.h"

namespace gpudirect_tcpxd {
std::unique_ptr<EventHandlerInterface> TcpSendEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  return std::make_unique<TcpSendEventHandler>(thread_id, socket, message_size,
                                               do_validation);
}

std::unique_ptr<EventHandlerInterface> TcpReceiveEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  return std::make_unique<TcpReceiveEventHandler>(thread_id, socket,
                                                  message_size, do_validation);
}

std::unique_ptr<EventHandlerInterface> TcpSendTcpDirectEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  return std::make_unique<TcpSendTcpDirectEventHandler>(
      thread_id, socket, message_size, do_validation);
}

std::unique_ptr<EventHandlerInterface>
TcpReceiveTcpDirectEventHandlerFactory::New(std::string thread_id, int socket,
                                            size_t message_size,
                                            bool do_validation) {
  return std::make_unique<TcpReceiveTcpDirectEventHanlder>(
      thread_id, socket, message_size, do_validation);
}

std::unique_ptr<EventHandlerInterface> DmabufSendEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  std::unique_ptr<CudaDmabufGpuPageAllocator> page_allocator =
      std::make_unique<CudaDmabufGpuPageAllocator>(gpu_pci_addr_,
                                                   nic_pci_addr_);
  return std::make_unique<GpuSendEventHandler>(thread_id, socket, message_size,
                                               do_validation,
                                               std::move(page_allocator));
}

std::unique_ptr<EventHandlerInterface> GpuReceiveEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  return std::make_unique<GpuReceiveEventHandler>(
      thread_id, socket, message_size, do_validation, gpu_pci_addr_);
}

// Dummy classes for testing

class DummyGpuPageAllocator : public GpuPageAllocatorInterface {
 public:
  void AllocatePage(size_t pool_size, unsigned long *msg_id,
                    bool *success) override {
    *msg_id = 0;
    *success = true;
  }
  void FreePage(unsigned long id) {}
  CUdeviceptr GetGpuMem(unsigned long id) override { return 0; }
  int GetGpuMemFd(unsigned long id) override { return -1; }
  void Reset() override {}
};

std::unique_ptr<EventHandlerInterface> DummyPageGpuSendEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  std::unique_ptr<DummyGpuPageAllocator> page_allocator =
      std::make_unique<DummyGpuPageAllocator>();
  return std::make_unique<GpuSendEventHandler>(thread_id, socket, message_size,
                                               do_validation,
                                               std::move(page_allocator));
}

std::unique_ptr<EventHandlerInterface> GpuSendMissFlagEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  std::unique_ptr<CudaDmabufGpuPageAllocator> page_allocator =
      std::make_unique<CudaDmabufGpuPageAllocator>(gpu_pci_addr_,
                                                   nic_pci_addr_);
  return std::make_unique<GpuSendEventHandlerMissFlag>(
      thread_id, socket, message_size, do_validation,
      std::move(page_allocator));
}

std::unique_ptr<EventHandlerInterface> GpuSendMixTcpEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  std::unique_ptr<CudaDmabufGpuPageAllocator> page_allocator =
      std::make_unique<CudaDmabufGpuPageAllocator>(gpu_pci_addr_,
                                                   nic_pci_addr_);
  return std::make_unique<GpuSendEventHandlerMixTcp>(
      thread_id, socket, message_size, do_validation,
      std::move(page_allocator));
}

std::unique_ptr<EventHandlerInterface> GpuSendOobEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  std::unique_ptr<CudaDmabufGpuPageAllocator> page_allocator =
      std::make_unique<CudaDmabufGpuPageAllocator>(gpu_pci_addr_,
                                                   nic_pci_addr_);
  return std::make_unique<GpuSendOobEventHandler>(thread_id, socket,
                                                  message_size, do_validation,
                                                  std::move(page_allocator));
}

std::unique_ptr<EventHandlerInterface>
GpuReceiveMissFlagEventHandlerFactory::New(std::string thread_id, int socket,
                                           size_t message_size,
                                           bool do_validation) {
  return std::make_unique<GpuReceiveMissFlagEventHandler>(
      thread_id, socket, message_size, do_validation, gpu_pci_addr_);
}

std::unique_ptr<EventHandlerInterface> GpuReceiveMixTcpEventHandlerFactory::New(
    std::string thread_id, int socket, size_t message_size,
    bool do_validation) {
  return std::make_unique<GpuReceiveMixTcpEventHandler>(
      thread_id, socket, message_size, do_validation, gpu_pci_addr_);
}

std::unique_ptr<EventHandlerInterface>
GpuReceiveTokenFreeEventHandlerFactory::New(std::string thread_id, int socket,
                                            size_t message_size,
                                            bool do_validation) {
  return std::make_unique<GpuReceiveTokenFreeEventHandler>(
      thread_id, socket, message_size, do_validation, gpu_pci_addr_);
}

std::unique_ptr<EventHandlerInterface>
GpuReceiveNoTokenFreeEventHandlerFactory::New(std::string thread_id, int socket,
                                              size_t message_size,
                                              bool do_validation) {
  return std::make_unique<GpuReceiveNoTokenFreeEventHandler>(
      thread_id, socket, message_size, do_validation, gpu_pci_addr_);
}

std::unique_ptr<EventHandlerFactoryInterface> EventHandlerFactorySelector(
    std::string event_handler_name, std::string gpu_pci_addr,
    std::string nic_pci_addr) {
  if (event_handler_name == "gpu_receive") {
    return std::make_unique<GpuReceiveEventHandlerFactory>(gpu_pci_addr);
  }
  if (event_handler_name == "gpu_receive_dummy_pci") {
    return std::make_unique<GpuReceiveEventHandlerFactory>(
        "random_string_not_gpu_pci");
  }
  if (event_handler_name == "gpu_receive_miss_flag") {
    return std::make_unique<GpuReceiveMissFlagEventHandlerFactory>(
        gpu_pci_addr);
  }
  if (event_handler_name == "gpu_receive_mix_tcp") {
    return std::make_unique<GpuReceiveMixTcpEventHandlerFactory>(gpu_pci_addr);
  }
  if (event_handler_name == "gpu_receive_token_free") {
    return std::make_unique<GpuReceiveTokenFreeEventHandlerFactory>(
        gpu_pci_addr);
  }
  if (event_handler_name == "gpu_receive_no_token_free") {
    return std::make_unique<GpuReceiveNoTokenFreeEventHandlerFactory>(
        gpu_pci_addr);
  }
  if (event_handler_name == "gpu_send") {
    return std::make_unique<DmabufSendEventHandlerFactory>(gpu_pci_addr,
                                                           nic_pci_addr);
  }
  if (event_handler_name == "gpu_sender_dummy_page") {
    return std::make_unique<DummyPageGpuSendEventHandlerFactory>();
  }
  if (event_handler_name == "gpu_send_miss_flag") {
    return std::make_unique<GpuSendMissFlagEventHandlerFactory>(gpu_pci_addr,
                                                                nic_pci_addr);
  }
  if (event_handler_name == "gpu_send_mix_tcp") {
    return std::make_unique<GpuSendMixTcpEventHandlerFactory>(gpu_pci_addr,
                                                              nic_pci_addr);
  }
  if (event_handler_name == "gpu_send_oob") {
    return std::make_unique<GpuSendOobEventHandlerFactory>(gpu_pci_addr,
                                                           nic_pci_addr);
  }
  if (event_handler_name == "tcp_receive") {
    return std::make_unique<TcpReceiveEventHandlerFactory>();
  }
  if (event_handler_name == "tcp_send") {
    return std::make_unique<TcpSendEventHandlerFactory>();
  }
  if (event_handler_name == "tcp_receive_tcp_direct") {
    return std::make_unique<TcpReceiveTcpDirectEventHandlerFactory>();
  }
  if (event_handler_name == "tcp_send_tcp_direct") {
    return std::make_unique<TcpSendTcpDirectEventHandlerFactory>();
  }
  return nullptr;
}
}  // namespace gpudirect_tcpxd
