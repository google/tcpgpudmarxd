#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_FLOW_STEER_NTUPLE_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_FLOW_STEER_NTUPLE_

#include <netinet/in.h>
#include <stdint.h>

namespace tcpdirect {

struct FlowSteerNtuple {
  uint32_t flow_type;
  union {
    struct {
      struct sockaddr_in src_sin;
      struct sockaddr_in dst_sin;
    };
    struct {
      struct sockaddr_in6 src_sin6;
      struct sockaddr_in6 dst_sin6;
    };
  };
};
}  // namespace tcpdirect
#endif
