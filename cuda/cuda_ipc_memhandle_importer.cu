#include <absl/log/check.h>
#include <absl/status/statusor.h>

#include <string>

#include "code.pb.h"
#include "cuda/cuda_ipc_memhandle.cuh"
#include "cuda/cuda_ipc_memhandle_importer.cuh"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {
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

  if (!resp.value().has_proto()) {
    return absl::NotFoundError("Response proto not found");
  }

  if (resp.value().proto().status().code() != google::rpc::Code::OK) {
    return absl::Status(absl::StatusCode(resp.value().proto().status().code()),
                        resp.value().proto().status().message());
  }

  if (!resp.value().proto().has_raw_bytes()) {
    return absl::NotFoundError("Memhandle not found in response proto");
  }

  return std::unique_ptr<GpuPageHandleInterface>(
      new CudaIpcMemhandle(resp.value().proto().raw_bytes()));
}
}  // namespace gpudirect_tcpxd
