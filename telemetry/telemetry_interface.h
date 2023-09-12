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

 private:
  std::thread worker_;
  std::atomic<bool> stopped_;
};

}  // namespace gpudirect_tcpxd

#endif