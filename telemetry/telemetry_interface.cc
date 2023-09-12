#include "telemetry/telemetry_interface.h"

#include <grpcpp/create_channel.h>

namespace gpudirect_tcpxd {

TelemetryInterface::TelemetryInterface() {}

TelemetryInterface::~TelemetryInterface() {
  stopped_ = true;
  worker_.join();
}

void TelemetryInterface::Start() {
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