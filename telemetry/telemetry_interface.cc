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

#include "telemetry/telemetry_interface.h"

#include <absl/flags/flag.h>
#include <grpcpp/create_channel.h>

ABSL_FLAG(bool, enable_rx_manager_telemetry, false,
          "Enable rx manager telemetry");

namespace gpudirect_tcpxd {

TelemetryInterface::TelemetryInterface() : started_(false), stopped_(false) {}

TelemetryInterface::~TelemetryInterface() {
  stopped_ = true;
  if (worker_.joinable()) {
    worker_.join();
  }
}

void TelemetryInterface::Start() {
  if (std::atomic_exchange(&started_, true)) {
    return;
  }

  if (!absl::GetFlag(FLAGS_enable_rx_manager_telemetry)) {
    return;
  }

  auto channel = grpc::CreateChannel("unix:///tmp/rx_buff_telemetry",
                                     grpc::InsecureChannelCredentials());
  stub_ = gpudirect_tcpxd::RxBufferTelemetryProxy::NewStub(channel);

  worker_ = std::thread([this]() {
    while (!stopped_) {
      PerSecondCounting();
      ReportTelemetry();
      absl::SleepFor(absl::Seconds(1));
    }
  });
}

}  // namespace gpudirect_tcpxd