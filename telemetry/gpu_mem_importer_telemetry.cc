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

#include "telemetry/gpu_mem_importer_telemetry.h"
#include <absl/log/log.h>

#include "include/unix_socket_connection.h"
#include "telemetry/proto/rx_manager_telemetry.pb.h"

namespace gpudirect_tcpxd {
GpuMemImporterTelemetry::GpuMemImporterTelemetry()
    : ipc_success_(0),
      ipc_failure_(0),
      import_success_(0),
      import_failure_(0) {}

void GpuMemImporterTelemetry::IncrementIpcSuccess() {
  absl::MutexLock lock(&mu_);
  ipc_success_ += 1;
}

void GpuMemImporterTelemetry::IncrementIpcFailure() {
  absl::MutexLock lock(&mu_);
  ipc_failure_ += 1;
}

void GpuMemImporterTelemetry::IncrementIpcFailureAndCause(
    std::string install_failure_cause) {
  absl::MutexLock lock(&mu_);
  ipc_failure_cause_map_[install_failure_cause]++;
}

void GpuMemImporterTelemetry::IncrementImportSuccess() {
  absl::MutexLock lock(&mu_);
  import_success_ += 1;
}

void GpuMemImporterTelemetry::IncrementImportFailure() {
  absl::MutexLock lock(&mu_);
  import_failure_ += 1;
}

void GpuMemImporterTelemetry::IncrementImportFailureAndCause(
    std::string install_failure_cause) {
  absl::MutexLock lock(&mu_);
  import_failure_cause_map_[install_failure_cause]++;
}

void GpuMemImporterTelemetry::UpdateImportType(std::string import_type) {
  absl::MutexLock lock(&mu_);
  importer_type_map_[import_type]++;
}

void GpuMemImporterTelemetry::AddLatency(absl::Duration latency) {
  absl::MutexLock lock(&mu_);
  if (latency > max_latency_) {
    max_latency_ = latency;
  }
  total_latency_sample_++;
  sum_latency_ += latency;
}

void GpuMemImporterTelemetry::ReportTelemetry() {
  GpuMemImporterTelemetryProto proto;

  {
    absl::MutexLock lock(&mu_);
    auto importer_ipc_install_failure_cause =
        proto.add_importer_ipc_install_failure_cause();
    for (auto i : ipc_failure_cause_map_) {
      importer_ipc_install_failure_cause
          ->set_importer_ipc_install_failure_cause(i.first);
      importer_ipc_install_failure_cause->set_count(i.second);
    }
    proto.set_ipc_success(ipc_success_);
    proto.set_ipc_failure(ipc_failure_);

    auto importer_import_install_failure_cause =
        proto.add_importer_import_install_failure_cause();
    for (auto i : ipc_failure_cause_map_) {
      importer_import_install_failure_cause->set_import_install_failure_cause(
          i.first);
      importer_import_install_failure_cause->set_count(i.second);
    }
    proto.set_import_success(import_success_);
    proto.set_import_failure(import_failure_);

    proto.set_avg_latency_ms(
        absl::ToDoubleMilliseconds(sum_latency_ / total_latency_sample_));
    proto.set_max_latency_ms(absl::ToDoubleMilliseconds(max_latency_));

    auto importer_type = proto.add_import_type();
    for (auto i : importer_type_map_) {
      importer_type->set_importer_type(i.first);
      importer_type->set_count(i.second);
    }
  }

  if (!stub_) {
    LOG(ERROR) << "Telemetry stub is not initialized";
    return;
  }

  grpc::ClientContext context;
  gpudirect_tcpxd::ReportStatus response;
  auto status = stub_->ReportGpuMemImporterMetrics(&context, proto, &response);
  if (!status.ok()) {
    LOG(ERROR) << "Report failed " << status.error_message();
  }
}
}  // namespace gpudirect_tcpxd