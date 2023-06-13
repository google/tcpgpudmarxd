#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_EVENT_HANDLER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_EVENT_HANDLER_H_

#include <string>

namespace tcpdirect {
class EventHandlerInterface {
 public:
  virtual ~EventHandlerInterface() = default;
  virtual unsigned InterestedEvents() = 0;
  virtual bool HandleEvents(unsigned events) = 0;
  virtual double GetRxBytes() = 0;
  virtual double GetTxBytes() = 0;
  virtual std::string Error() = 0;
};
}  // namespace tcpdirect

#endif
