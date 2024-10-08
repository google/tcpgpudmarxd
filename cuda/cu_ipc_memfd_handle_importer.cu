/*
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/memory/memory.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>

#include <memory>
#include <string>

#include "code.pb.h"
#include "cuda/common.cuh"
#include "cuda/cu_ipc_memfd_handle.cuh"
#include "cuda/cu_ipc_memfd_handle_importer.cuh"
#include "include/ipc_gpumem_fd_metadata.h"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"
#include "telemetry/gpu_mem_importer_telemetry.h"

namespace gpudirect_tcpxd {

absl::StatusOr<std::unique_ptr<GpuPageHandleInterface>>
CuIpcMemfdHandleImporter::Import(const std::string& prefix,
                                 const std::string& gpu_pci_addr) {
  static GpuMemImporterTelemetry telemetry;
  telemetry.Start(); // No-op if already started

  absl::Time start = absl::Now();
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
  if (!resp.status().ok()) {
    telemetry.IncrementIpcFailure();
    telemetry.IncrementIpcFailureAndCause(resp.status().ToString());
    return resp.status();
  }
  if (!resp.value().has_fd() || resp.value().fd() < 0) {
    telemetry.IncrementIpcFailure();
    telemetry.IncrementIpcFailureAndCause("Fd Not found");
    return absl::NotFoundError("Not found");
  }
  telemetry.IncrementIpcSuccess();

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
    telemetry.IncrementImportFailure();
    telemetry.IncrementImportFailureAndCause("Response proto not found");

    return absl::NotFoundError("Response proto not found");
  }

  if (resp_metadata->proto().status().code() != google::rpc::Code::OK) {
    telemetry.IncrementImportFailure();
    telemetry.IncrementImportFailureAndCause(
        resp_metadata->proto().status().message());
    return absl::Status(
        absl::StatusCode(resp_metadata->proto().status().code()),
        resp_metadata->proto().status().message());
  }

  if (!resp_metadata.value().proto().has_raw_bytes()) {
        telemetry.IncrementImportFailure();
    telemetry.IncrementImportFailureAndCause(
        "Memhandle not found in response proto");
    return absl::NotFoundError("Memhandle not found in response proto");
  }

  memcpy((void*)&gpumem_fd_metadata,
         (void*)resp_metadata.value().proto().raw_bytes().data(),
         resp_metadata.value().proto().raw_bytes().size());

  int dev_id;
  CUDA_ASSERT_SUCCESS(cudaDeviceGetByPCIBusId(&dev_id, gpu_pci_addr.c_str()));
  telemetry.IncrementImportSuccess();
  telemetry.AddLatency(absl::Now()-start);
  telemetry.UpdateImportType("Cuda Fd Mem Handle");
  return std::unique_ptr<GpuPageHandleInterface>(
      new CuIpcMemfdHandle(resp.value().fd(), dev_id, gpumem_fd_metadata.size,
                           gpumem_fd_metadata.align));
}
}  // namespace gpudirect_tcpxd
