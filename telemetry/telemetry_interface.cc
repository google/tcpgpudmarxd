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