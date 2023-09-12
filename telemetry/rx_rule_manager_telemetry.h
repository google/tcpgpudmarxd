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

#ifndef TCPGPUDMAD_RX_BUF_TELEMETRY_HELPER_H_
#define TCPGPUDMAD_RX_BUF_TELEMETRY_HELPER_H_

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>
#include <absl/time/time.h>

#include <cstddef>
#include <optional>
#include <string>
#include <thread>

#include "telemetry/telemetry_interface.h"

namespace gpudirect_tcpxd {
class FlowSteerRuleManagerTelemetry: public TelemetryInterface {
 public:
  FlowSteerRuleManagerTelemetry();
  ~FlowSteerRuleManagerTelemetry();
  void IncrementInstallSuccess();
  void IncrementInstallFailure();
  void IncrementUninstallSuccess();
  void IncrementFailureAndCause(std::string install_failure_cause);
  void IncrementRulesInstalledOnQueues(size_t queue);
  void AddLatency(absl::Duration latency);
  void IncrementRequests();

protected:
  void PerSecondCounting() override;
  void ReportTelemetry() override;

 private:
  size_t install_success_;
  size_t install_failure_;
  size_t uninstall_success_;
  absl::flat_hash_map<std::string, size_t> failure_cause_map_;
  absl::flat_hash_map<size_t, size_t> queues_rules_map_;
  size_t total_latency_sample_;
  absl::Duration sum_latency_;
  absl::Duration max_latency_;
  absl::Time start_time_;
  size_t total_requests_sample_;
  size_t prev_sec_count_;
  size_t max_count_diff_;
  absl::Mutex mu_;
};
};  // namespace gpudirect_tcpxd

#endif  // TCPGPUDMAD_RX_BUF_TELEMETRY_HELPER_H_