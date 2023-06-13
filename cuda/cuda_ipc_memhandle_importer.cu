#include "cuda/cuda_ipc_memhandle_importer.cu.h"

#include <string>

#include "cuda/cuda_ipc_memhandle.cu.h"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.proto.h"
#include <absl/status/statusor.h>

namespace tcpdirect {
absl::StatusOr<std::unique_ptr<GpuPageHandleInterface>>
CudaIpcMemhandleImporter::Import(const std::string& prefix,
                                 const std::string& gpu_pci_addr) {
  UnixSocketClient gpumem_by_gpu_pci_client(prefix + "/get_gpu_by_gpu_pci");
  PCHECK(gpumem_by_gpu_pci_client.Connect().ok());
  UnixSocketMessage req;
  UnixSocketProto* mutable_proto = req.mutable_proto();
  mutable_proto->set_raw_bytes(gpu_pci_addr);
  gpumem_by_gpu_pci_client.Send(req);
  absl::StatusOr<UnixSocketMessage> resp = gpumem_by_gpu_pci_client.Receive();
  if (!resp.status().ok()) return resp.status();
  if (resp.value().has_proto() && resp.value().proto().has_raw_bytes()) {
    return std::unique_ptr<GpuPageHandleInterface>(
        new CudaIpcMemhandle(resp.value().proto().raw_bytes()));
  }
  return absl::NotFoundError("Not found");
}
}  // namespace tcpdirect
