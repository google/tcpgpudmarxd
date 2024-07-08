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

#ifndef TCPGPUDMAD_RX_TELEMETRY_BASE_H_
#define TCPGPUDMAD_RX_TELEMETRY_BASE_H_

#include <absl/time/clock.h>

#include <atomic>
#include <memory>
#include <thread>

#include "telemetry/proto/rx_manager_telemetry.grpc.pb.h"

namespace gpudirect_tcpxd {
class TelemetryInterface {
 public:
  TelemetryInterface() ;
  virtual ~TelemetryInterface();
  void Start();

 protected:
  virtual void PerSecondCounting() {};
  virtual void ReportTelemetry() = 0;
  std::unique_ptr<gpudirect_tcpxd::RxBufferTelemetryProxy::Stub> stub_;
  std::atomic<bool> started_;

 private:
  std::thread worker_;
  std::atomic<bool> stopped_;
};

}  // namespace gpudirect_tcpxd

#endif