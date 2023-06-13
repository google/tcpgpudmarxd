#include "experimental/users/chechenglin/tcpgpudmad/benchmark/benchmark_common.h"

#include <arpa/inet.h>
#include <ifaddrs.h>

#include <string>
#include <string_view>
#include <unordered_map>

#include "base/logging.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/str_format.h"

namespace tcpdirect {

namespace {
inline constexpr std::string_view kIpv6LinkLocalPrefix = "fe80::";
}

void GetNicPciAddrToIpMap(NicPciAddrToIpMap* nic_pci_addr_to_ip_addr,
                          const Dictionary& pci_to_ifname) {
  for (const auto& kv : pci_to_ifname) {
    union SocketAddress addr;
    struct ifaddrs* ifap;
    struct ifaddrs* ifa;

    const auto& if_name = kv.second;

    getifaddrs(&ifap);
    for (ifa = ifap; ifa; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET6) {
        struct sockaddr_in6* sa = (struct sockaddr_in6*)ifa->ifa_addr;
        if (std::string(ifa->ifa_name) == if_name) {
          char addr_buf[256];
          const char* addr_str_arr =
              inet_ntop(sa->sin6_family, (void*)&sa->sin6_addr, addr_buf, 256);
          std::string addr_str(addr_str_arr);
          if (!absl::StartsWithIgnoreCase(addr_str, kIpv6LinkLocalPrefix)) {
            addr.sin6 = *sa;
          }
        }
      }
    }
    freeifaddrs(ifap);

    (*nic_pci_addr_to_ip_addr)[kv.first] = addr;
  }

  for (const auto& kv : *nic_pci_addr_to_ip_addr) {
    char addr_buf[256];
    const char* addr_str_arr =
        inet_ntop(kv.second.sin6.sin6_family, (void*)&kv.second.sin6.sin6_addr,
                  addr_buf, 256);
    std::string addr_str(addr_str_arr);
    LOG(INFO) << absl::StrFormat("NIC %s: addr %s", kv.first, addr_str);
  }
}

union SocketAddress GetIpv6FromIfName(const std::string& ifname) {
  static std::unordered_map<std::string, union SocketAddress>* ifname_ip6 =
      new std::unordered_map<std::string, union SocketAddress>();
  union SocketAddress addr;

  if (ifname_ip6->empty()) {
    struct ifaddrs* ifap;
    struct ifaddrs* ifa;

    getifaddrs(&ifap);
    for (ifa = ifap; ifa; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr) {
        if (ifa->ifa_addr->sa_family == AF_INET6) {
          struct sockaddr_in6* sa = (struct sockaddr_in6*)ifa->ifa_addr;
          std::string ifname = std::string(ifa->ifa_name);
          char addr_buf[256];
          const char* addr_str_arr =
              inet_ntop(sa->sin6_family, (void*)&sa->sin6_addr, addr_buf, 256);
          std::string addr_str(addr_str_arr);
          if (!absl::StartsWithIgnoreCase(addr_str, kIpv6LinkLocalPrefix)) {
            addr.sin6 = *sa;
          }
          ifname_ip6->emplace(ifname, addr);
        }
      }
    }
    freeifaddrs(ifap);
  }

  if (ifname_ip6->find(ifname) != ifname_ip6->end()) {
    return ifname_ip6->at(ifname);
  }
  return addr;
}

}  // namespace tcpdirect
