#include <absl/log/log.h>
#include <absl/status/statusor.h>
#include <fcntl.h>
#include <sys/socket.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>

#include "cuda/common.cuh"
#include "cuda/cuda_ipc_memhandle_importer.cuh"
#include "include/unix_socket_client.h"
#include "machine_test/cuda/gpu_receive_common.cuh"
#include "proto/unix_socket_message.pb.h"

#define USE_UNIX_SOCKET

namespace gpudirect_tcpxd {

namespace {
constexpr long long kMaxOptLen{4096};
constexpr long long kMaxTokensToFree = kMaxOptLen / sizeof(TokenT);
}  // namespace

CUdeviceptr gpumem_import(const std::string& gpu_pci_addr,
                          const std::string& thread_id) {
  // The original benchmark implementation let multiple workers share the same
  // GPU rx buffer. Workers assigned the same gpu_pci_addr share the same GPU
  // rx buffer.

  static std::mutex rxbuf_mutex;
  static std::unordered_map<std::string,
                            std::unique_ptr<GpuPageHandleInterface>>
      gpu_addr_rx_buff_map;
  {
    std::lock_guard<std::mutex> lk(rxbuf_mutex);
    if (gpu_addr_rx_buff_map.find(gpu_pci_addr) != gpu_addr_rx_buff_map.end()) {
      return gpu_addr_rx_buff_map[gpu_pci_addr]->GetGpuMem();
    }
    LOG(INFO) << thread_id << ": Importing rx buf from: unix socket succeed: "
              << gpu_pci_addr;

    absl::StatusOr<std::unique_ptr<GpuPageHandleInterface>> gpu_page_handle =
        CudaIpcMemhandleImporter::Import("/tmp", gpu_pci_addr);
    if (!gpu_page_handle.status().ok()) {
      LOG(ERROR) << thread_id
                 << "Failed to import rx buf for : " << gpu_pci_addr << " "
                 << gpu_page_handle.status();
      return 0;
    }
    gpu_addr_rx_buff_map[gpu_pci_addr] = std::move(gpu_page_handle.value());
    return gpu_addr_rx_buff_map[gpu_pci_addr]->GetGpuMem();
  }
}

int FreeRxPages(const std::vector<TokenT>& tokens_to_free, int socket) {
  long long total = tokens_to_free.size();
  long long offset = 0;
  int ret = 0;
  while (total > 0) {
    size_t this_batch = std::min(kMaxTokensToFree, total);
    size_t optlen = this_batch * sizeof(TokenT);
    ret =
        setsockopt(socket, SOL_SOCKET, SO_DEVMEM_DONTNEED,
                   tokens_to_free.data() + offset, this_batch * sizeof(TokenT));
    if (ret != 0 && errno != EINTR) {
      PLOG(ERROR) << "Error while freeing rx p2p pages";
      return ret;
    }
    total -= this_batch;
    offset += optlen;
  }
  return ret;
}

}  // namespace gpudirect_tcpxd
