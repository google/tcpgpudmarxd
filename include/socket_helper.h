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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_SOCKET_HELPER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_SOCKET_HELPER_H_

#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tcpdirect {

union SocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

struct NetifInfo {
  std::string ifname;
  union SocketAddress addr;
  uint16_t dbdf[4];
  bool has_pcie_addr;
};

int CreateTcpSocket(int address_family);
void SetReuseAddr(int fd);
void EnableTcpZeroCopy(int fd);

void SetAddressPort(union SocketAddress* addr, uint16_t port);
uint16_t GetAddressPort(const union SocketAddress* addr);
union SocketAddress AddressFromStr(const std::string& str);
std::string AddressToStr(const union SocketAddress* addr);

void BindAndListen(int fd, union SocketAddress* addr, int backlog);
void BindAndListen(int fd, struct sockaddr* addr, socklen_t addrlen,
                   int backlog);
void BindAndListen(int fd, struct sockaddr_in* bind_addr, int backlog);
void BindAndListen(int fd, struct sockaddr_in6* bind_addr, int backlog);

void ConnectWithRetry(int fd, union SocketAddress* addr, int max_retry = 10);
void ConnectWithRetry(int fd, struct sockaddr* addr, socklen_t addrlen,
                      int max_retry = 10);
void ConnectWithRetry(int fd, struct sockaddr_in* addr, int max_retry = 10);
void ConnectWithRetry(int fd, struct sockaddr_in6* addr, int max_retry = 10);

void AcceptConnectionAndSendHpnAddress(
    union SocketAddress* server_addr,
    const std::vector<union SocketAddress>& server_hpn_addresses);
void ConnectAndReceiveHpnAddress(
    union SocketAddress* server_addr,
    std::vector<union SocketAddress>* server_hpn_addresses);

int ReadNicPci(const char* ifname, uint16_t* dbdf);

void DiscoverNetif(std::vector<NetifInfo>& nic_info);

template <typename F>
void DiscoverNetif(std::vector<NetifInfo>& nic_info, F&& filter) {
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    if (!filter(interface)) {
      continue;
    }
    int family = interface->ifa_addr->sa_family;
    std::string ifname = std::string(interface->ifa_name);
    struct NetifInfo info;
    info.ifname = ifname;
    if (family == AF_INET) {
      memcpy(&info.addr.sin, interface->ifa_addr, sizeof(sockaddr_in));
    } else {
      memcpy(&info.addr.sin6, interface->ifa_addr, sizeof(sockaddr_in6));
    }
    int ret = ReadNicPci(ifname.c_str(), info.dbdf);
    if (ret == 0) {
      info.has_pcie_addr = true;
    } else {
      info.has_pcie_addr = false;
    }
    nic_info.emplace_back(info);
  }
}
}  // namespace tcpdirect

#endif /* _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_SOCKET_HELPER_H_ */
