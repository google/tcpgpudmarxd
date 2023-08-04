#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>

#include <string>

#include "code.pb.h"
#include "cuda/common.cuh"
#include "cuda/cu_ipc_memfd_handle.cuh"
#include "cuda/cu_ipc_memfd_handle_importer.cuh"
#include "include/ipc_gpumem_fd_metadata.h"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {
absl::StatusOr<std::unique_ptr<GpuPageHandleInterface>>
CuIpcMemfdHandleImporter::Import(const std::string& prefix,
                                 const std::string& gpu_pci_addr) {
  IpcGpuMemFdMetadata gpumem_fd_metadata;
  // fetch ipc shareable fd
  UnixSocketClient gpumem_fd_by_gpu_pci_client(
      absl::StrFormat("%s/get_gpu_fd_%s", prefix, gpu_pci_addr));
  PCHECK(gpumem_fd_by_gpu_pci_client.Connect().ok());
  UnixSocketMessage req;
  UnixSocketProto* req_mutable_proto = req.mutable_proto();
  req_mutable_proto->set_raw_bytes(gpu_pci_addr);
  gpumem_fd_by_gpu_pci_client.Send(req);
  absl::StatusOr<UnixSocketMessage> resp =
      gpumem_fd_by_gpu_pci_client.Receive();
  if (!resp.status().ok()) return resp.status();
  if (!resp.value().has_fd() || resp.value().fd() < 0) {
    return absl::NotFoundError("Not found");
  }

  // fetch gpu memory metadata
  UnixSocketClient gpumem_metadata_by_gpu_pci_client(
      absl::StrFormat("%s/get_gpu_metadata_%s", prefix, gpu_pci_addr));
  PCHECK(gpumem_metadata_by_gpu_pci_client.Connect().ok());
  UnixSocketMessage req_metadata;
  UnixSocketProto* md_mutable_proto = req_metadata.mutable_proto();
  md_mutable_proto->set_raw_bytes(gpu_pci_addr);
  gpumem_metadata_by_gpu_pci_client.Send(req_metadata);
  absl::StatusOr<UnixSocketMessage> resp_metadata =
      gpumem_metadata_by_gpu_pci_client.Receive();
  PCHECK(resp_metadata.status().ok());

  if (!resp_metadata.value().has_proto()) {
    return absl::NotFoundError("Response proto not found");
  }

  if (resp_metadata->proto().status().code() != google::rpc::Code::OK) {
    return absl::Status(
        absl::StatusCode(resp_metadata->proto().status().code()),
        resp_metadata->proto().status().message());
  }

  if (!resp_metadata.value().proto().has_raw_bytes()) {
    return absl::NotFoundError("Memhandle not found in response proto");
  }

  memcpy((void*)&gpumem_fd_metadata,
         (void*)resp_metadata.value().proto().raw_bytes().data(),
         resp_metadata.value().proto().raw_bytes().size());

  int dev_id;
  CUDA_ASSERT_SUCCESS(cudaDeviceGetByPCIBusId(&dev_id, gpu_pci_addr.c_str()));
  return std::unique_ptr<GpuPageHandleInterface>(
      new CuIpcMemfdHandle(resp.value().fd(), dev_id, gpumem_fd_metadata.size,
                           gpumem_fd_metadata.align));
}
}  // namespace gpudirect_tcpxd
