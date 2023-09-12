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

#include "telemetry/gpu_mem_exporter_telemetry.h"

#include <absl/log/log.h>
#include <absl/time/clock.h>

#include "telemetry/proto/rx_manager_telemetry.pb.h"

namespace gpudirect_tcpxd {
GpuMemExporterTelemetry::GpuMemExporterTelemetry()
    : ipc_success_(0),
      ipc_failure_(0),
      total_latency_sample_(0),
      start_time_(absl::Now()),
      total_requests_sample_(0),
      prev_sec_count_(0),
      max_count_diff_(0) {}

void GpuMemExporterTelemetry::IncrementIpcSuccess() {
  absl::MutexLock lock(&mu_);
  ipc_success_ += 1;
}

void GpuMemExporterTelemetry::IncrementIpcFailure() {
  absl::MutexLock lock(&mu_);
  ipc_failure_ += 1;
}

void GpuMemExporterTelemetry::IncrementIpcFailureAndCause(
    std::string install_failure_cause) {
  absl::MutexLock lock(&mu_);
  ipc_failure_cause_map_[install_failure_cause]++;
}

void GpuMemExporterTelemetry::AddLatency(absl::Duration latency) {
  absl::MutexLock lock(&mu_);
  if (latency > max_latency_) {
    max_latency_ = latency;
  }
  total_latency_sample_++;
  sum_latency_ += latency;
}

void GpuMemExporterTelemetry::IncrementRequests() {
  absl::MutexLock lock(&mu_);
  total_requests_sample_++;
}

void GpuMemExporterTelemetry::PerSecondCounting() {
  absl::MutexLock lock(&mu_);
  auto diff = total_requests_sample_ - prev_sec_count_;
  if (diff > max_count_diff_) {
    max_count_diff_ = diff;
  }
  prev_sec_count_ = total_latency_sample_;
}

void GpuMemExporterTelemetry::ReportTelemetry() {
  GpuMemExporterTelemetryProto proto;

  {
    absl::MutexLock lock(&mu_);
    auto exporter_ipc_install_failure_cause =
        proto.add_exporter_ipc_install_failure_cause();
    for (auto i : ipc_failure_cause_map_) {
      exporter_ipc_install_failure_cause
          ->set_exporter_ipc_install_failure_cause(i.first);
      exporter_ipc_install_failure_cause->set_count(i.second);
    }
    proto.set_ipc_success(ipc_success_);
    proto.set_ipc_failure(ipc_failure_);
    proto.set_avg_latency_ms(
        absl::ToDoubleMilliseconds(sum_latency_ / total_latency_sample_));
    proto.set_max_latency_ms(absl::ToDoubleMilliseconds(max_latency_));
    proto.set_avg_rps(total_requests_sample_ /
                      absl::ToDoubleSeconds(absl::Now() - start_time_));
    proto.set_max_rps(max_count_diff_);
  }

  if (!stub_) {
    LOG(ERROR) << "Telemetry stub is not initialized";
    return;
  }

  grpc::ClientContext context;
  gpudirect_tcpxd::ReportStatus response;
  auto status = stub_->ReportGpuMemExporterMetrics(&context, proto, &response);
  if (!status.ok()) {
    LOG(ERROR) << "Report failed " << status.error_message();
  }
}

}  // namespace gpudirect_tcpxd