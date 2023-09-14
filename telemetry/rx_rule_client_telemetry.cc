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

#include "telemetry/rx_rule_client_telemetry.h"

#include <absl/log/log.h>
#include <absl/time/time.h>
#include <grpc/grpc.h>
#include <grpcpp/create_channel.h>

#include <iterator>

#include "include/unix_socket_connection.h"
#include "telemetry/proto/rx_manager_telemetry.grpc.pb.h"
#include "telemetry/proto/rx_manager_telemetry.pb.h"

namespace gpudirect_tcpxd {
FlowSteerRuleClientTelemetry::FlowSteerRuleClientTelemetry()
    : install_success_(0), install_failure_(0), total_latency_sample_(0) {}

void FlowSteerRuleClientTelemetry::IncrementInstallSuccess() {
  absl::MutexLock lock(&mu_);
  install_success_ += 1;
}

void FlowSteerRuleClientTelemetry::IncrementInstallFailure() {
  absl::MutexLock lock(&mu_);
  install_success_ += 1;
}

void FlowSteerRuleClientTelemetry::IncrementFailureAndCause(
    std::string install_failure_cause) {
  absl::MutexLock lock(&mu_);
  failure_cause_map_[install_failure_cause]++;
}

void FlowSteerRuleClientTelemetry::AddLatency(absl::Duration latency) {
  absl::MutexLock lock(&mu_);
  if (latency > max_latency_) {
    max_latency_ = latency;
  }
  total_latency_sample_++;
  sum_latency_ += latency;
}

void FlowSteerRuleClientTelemetry::ReportTelemetry() {
  FlowSteerRuleClientTelemetryProto proto;
  {
    absl::MutexLock lock(&mu_);
    auto client_install_failure_cause =
        proto.add_client_install_failure_cause();
    for (auto i : failure_cause_map_) {
      client_install_failure_cause->set_install_failure_cause(i.first);
      client_install_failure_cause->set_count(i.second);
    }
    proto.set_install_failure(install_failure_);
    proto.set_install_success(install_success_);
    proto.set_avg_latency_ms(
        absl::ToDoubleMilliseconds(sum_latency_ / total_latency_sample_));
    proto.set_max_latency_ms(absl::ToDoubleMilliseconds(max_latency_));
  }

  if (!stub_) {
    LOG(ERROR) << "Telemetry stub is not initialized";
    return;
  }
  grpc::ClientContext context;
  gpudirect_tcpxd::ReportStatus response;
  auto status = stub_->ReportFlowSteerRuleClientMetrics(&context, proto, &response);
  if (!status.ok()) {
    LOG(ERROR) << "Report failed " << status.error_message();
  }
}
}  // namespace gpudirect_tcpxd