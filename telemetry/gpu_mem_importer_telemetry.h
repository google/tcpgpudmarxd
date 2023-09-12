/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TCPGPUDMAD_RX_GPU_MEM_IMPORTER_TELEMETRY_H_
#define TCPGPUDMAD_RX_GPU_MEM_IMPORTER_TELEMETRY_H_

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>
#include <absl/time/time.h>

#include <cstddef>
#include <optional>
#include <string>

#include "telemetry/telemetry_interface.h"

namespace gpudirect_tcpxd {
class GpuMemImporterTelemetry : public TelemetryInterface {
 public:
  GpuMemImporterTelemetry();
  void IncrementIpcSuccess();
  void IncrementIpcFailure();
  void IncrementIpcFailureAndCause(std::string install_failure_cause);
  void IncrementImportSuccess();
  void IncrementImportFailure();
  void IncrementImportFailureAndCause(std::string install_failure_cause);
  void UpdateImportType(std::string import_type);
  void AddLatency(absl::Duration latency);

 private:
  void ReportTelemetry() override;

  size_t ipc_success_;
  size_t ipc_failure_;
  absl::flat_hash_map<std::string, size_t> ipc_failure_cause_map_;
  size_t import_success_;
  size_t import_failure_;
  absl::flat_hash_map<std::string, size_t> import_failure_cause_map_;
  absl::flat_hash_map<std::string, size_t> importer_type_map_;
  absl::Duration sum_latency_;
  absl::Duration max_latency_;
  size_t total_latency_sample_;
  absl::Mutex mu_;
};
};  // namespace gpudirect_tcpxd

#endif  // TCPGPUDMAD_RX_GPU_MEM_IMPORTER_TELEMETRY_H_