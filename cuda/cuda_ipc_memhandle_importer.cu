#include <absl/log/check.h>
#include <absl/status/statusor.h>

#include <string>

#include "code.pb.h"
#include "cuda/cuda_ipc_memhandle.cuh"
#include "cuda/cuda_ipc_memhandle_importer.cuh"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"
#include "telemetry/gpu_mem_importer_telemetry.h"

namespace gpudirect_tcpxd {
absl::StatusOr<std::unique_ptr<GpuPageHandleInterface>>
CudaIpcMemhandleImporter::Import(const std::string& prefix,
                                 const std::string& gpu_pci_addr) {
  static GpuMemImporterTelemetry telemetry;
  telemetry.Start();  // No-op if already started

  absl::Time start = absl::Now();
  UnixSocketClient gpumem_by_gpu_pci_client(prefix + "/get_gpu_by_gpu_pci");
  PCHECK(gpumem_by_gpu_pci_client.Connect().ok());
  UnixSocketMessage req;
  UnixSocketProto* mutable_proto = req.mutable_proto();
  mutable_proto->set_raw_bytes(gpu_pci_addr);
  gpumem_by_gpu_pci_client.Send(req);
  absl::StatusOr<UnixSocketMessage> resp = gpumem_by_gpu_pci_client.Receive();
  if (!resp.status().ok()) {
    telemetry.IncrementImportFailure();
    telemetry.IncrementImportFailureAndCause(resp.status().ToString());
    return resp.status();
  }

  if (!resp.value().has_proto()) {
    telemetry.IncrementImportFailure();
    telemetry.IncrementImportFailureAndCause("Response proto not found");
    return absl::NotFoundError("Response proto not found");
  }

  if (resp.value().proto().status().code() != google::rpc::Code::OK) {
    telemetry.IncrementImportFailure();
    telemetry.IncrementImportFailureAndCause(resp->proto().status().message());
    return absl::Status(absl::StatusCode(resp.value().proto().status().code()),
                        resp.value().proto().status().message());
  }

  if (!resp.value().proto().has_raw_bytes()) {
    telemetry.IncrementImportFailure();
    telemetry.IncrementImportFailureAndCause(
        "Memhandle not found in response proto");
    return absl::NotFoundError("Memhandle not found in response proto");
  }

  telemetry.IncrementImportSuccess();
  telemetry.AddLatency(absl::Now() - start);
  telemetry.UpdateImportType("Cuda Mem Handle");
  return std::unique_ptr<GpuPageHandleInterface>(
      new CudaIpcMemhandle(resp.value().proto().raw_bytes()));
}
}  // namespace gpudirect_tcpxd
