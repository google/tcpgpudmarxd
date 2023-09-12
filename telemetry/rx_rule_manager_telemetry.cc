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

#include "telemetry/rx_rule_manager_telemetry.h"

#include <absl/log/log.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <grpc/grpc.h>

#include <thread>

#include "telemetry/proto/rx_manager_telemetry.grpc.pb.h"
#include "telemetry/proto/rx_manager_telemetry.pb.h"

namespace gpudirect_tcpxd {

FlowSteerRuleManagerTelemetry::FlowSteerRuleManagerTelemetry()
    : TelemetryInterface(),
      install_success_(0),
      install_failure_(0),
      uninstall_success_(0),
      total_latency_sample_(0),
      start_time_(absl::Now()),
      total_requests_sample_(0),
      prev_sec_count_(0),
      max_count_diff_(0)
{}

FlowSteerRuleManagerTelemetry::~FlowSteerRuleManagerTelemetry() {}

void FlowSteerRuleManagerTelemetry::IncrementInstallSuccess() {
  absl::MutexLock lock(&mu_);
  install_success_ += 1;
}

void FlowSteerRuleManagerTelemetry::IncrementInstallFailure() {
  absl::MutexLock lock(&mu_);
  install_failure_ += 1;
}

void FlowSteerRuleManagerTelemetry::IncrementUninstallSuccess() {
  absl::MutexLock lock(&mu_);
  uninstall_success_ += 1;
}

void FlowSteerRuleManagerTelemetry::IncrementFailureAndCause(
    std::string install_failure_cause) {
  absl::MutexLock lock(&mu_);
  failure_cause_map_[install_failure_cause]++;
}

void FlowSteerRuleManagerTelemetry::IncrementRulesInstalledOnQueues(
    size_t queue) {
  absl::MutexLock lock(&mu_);
  queues_rules_map_[queue]++;
}

void FlowSteerRuleManagerTelemetry::AddLatency(absl::Duration latency) {
  absl::MutexLock lock(&mu_);
  if (latency > max_latency_) {
    max_latency_ = latency;
  }
  total_latency_sample_++;
  sum_latency_ += latency;
}

void FlowSteerRuleManagerTelemetry::IncrementRequests() {
  absl::MutexLock lock(&mu_);
  total_requests_sample_++;
}

void FlowSteerRuleManagerTelemetry::PerSecondCounting() {
  absl::MutexLock lock(&mu_);
  auto diff = total_requests_sample_ - prev_sec_count_;
  if (diff > max_count_diff_) {
    max_count_diff_ = diff;
  }
  prev_sec_count_ = total_latency_sample_;
}

void FlowSteerRuleManagerTelemetry::ReportTelemetry() {
  FlowSteerRuleManagerTelemetryProto proto;
  {
    absl::MutexLock lock(&mu_);
    auto manager_isntall_failure_cause =
        proto.add_manager_install_failure_cause();
    for (auto i : failure_cause_map_) {
      manager_isntall_failure_cause->set_install_failure_cause(i.first);
      manager_isntall_failure_cause->set_count(i.second);
    }
    auto rules_in_queues = proto.add_rules_in_queues();
    for (auto i : queues_rules_map_) {
      rules_in_queues->set_queue(i.first);
      rules_in_queues->set_rules_installed(i.second);
    }
    proto.set_install_success(install_success_);
    proto.set_install_failure(install_failure_);
    proto.set_avg_latency_ms(
        absl::ToDoubleMilliseconds(sum_latency_ / total_latency_sample_));
    proto.set_max_latency_ms(absl::ToDoubleMilliseconds(max_latency_));
    proto.set_rules_in_use(install_success_ - uninstall_success_);
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
  auto status = stub_->ReportRxRuleManagerMetrics(&context, proto, &response);
  if (!status.ok()) {
    LOG(ERROR) << "Report failed " << status.error_message();
  }
}

}  // namespace gpudirect_tcpxd