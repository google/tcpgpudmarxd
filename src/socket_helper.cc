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

#include "include/socket_helper.h"

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <vector>

namespace gpudirect_tcpxd {

namespace {
static const std::string kIpv6LinkLocalPrefix = "fe80::";
static const std::vector<std::string> kIgnoredIf = {"lo"};
static const std::vector<std::string> kIgnoredIfPrefix = {"docker"};
}  // namespace

int CreateTcpSocket(int address_family) {
  int fd = socket(address_family, SOCK_STREAM, 0);
  PCHECK(fd >= 0);
  return fd;
}

int GetIncomingCpu(int fd) {
  int cpu = -1;
  socklen_t result_len = sizeof(cpu);
  if (getsockopt(fd, SOL_SOCKET, SO_INCOMING_CPU, &cpu, &result_len)) {
    LOG(FATAL) << "GetIncomingCpu: " << strerror(errno);
  }
  return cpu;
}

int GetSockNapiID(int fd) {
  int cpu = -1;
  socklen_t result_len = sizeof(cpu);
  if (getsockopt(fd, SOL_SOCKET, SO_INCOMING_NAPI_ID, &cpu, &result_len)) {
    LOG(FATAL) << "GetSockNapiID: " << strerror(errno);
  }
  return cpu;
}

void SetReuseAddr(int fd) {
  int opt = 1;
  int ret = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                       sizeof(opt));
  PCHECK(ret == 0);
}

void EnableTcpZeroCopy(int fd) {
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &opt, sizeof(opt))) {
    perror("EnableTcpZeroCopy: ");
    exit(EXIT_FAILURE);
  }
}

void SetTcpNoDelay(int sock) {
  int opt = 1;
  if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)&opt, sizeof(opt))) {
    LOG(FATAL) << "SetTcpNoDelay: " << strerror(errno);
  }
}

void GetNicNumaCores(cpu_set_t* cpuset, const std::string& ifname) {
  int num_cores = sysconf(_SC_NPROCESSORS_CONF);
  CHECK(num_cores > 0);
  for (int i = 0; i < num_cores; i++) {
    CPU_SET(i, cpuset);
  }
  char buf[256];
  ssize_t ret;
  std::unordered_set<int> cpu_idx_set;

  int numa_node = 0;
  std::string numa_node_path =
      absl::StrFormat("/sys/class/net/%s/device/numa_node", ifname);
  std::string numa_dir;
  int fd = open(numa_node_path.c_str(), O_RDONLY);
  if (fd < 0) {
    goto default_cpuset;
  }
  ret = read(fd, &buf, sizeof(buf) - 1);
  CHECK(ret >= 0);
  buf[ret] = 0;
  numa_node = std::strtol(buf, nullptr, 0);

  numa_dir = absl::StrFormat("/sys/devices/system/node/node%d/", numa_node);

  for (const auto& entry : std::filesystem::directory_iterator(numa_dir)) {
    if (!entry.is_directory()) continue;
    std::string basename = entry.path().filename();
    if (absl::StartsWith(basename, "cpu")) {
      for (int i = 3; i < basename.size(); i++) {
        if (!std::isdigit(basename[i])) {
          continue;
        }
      }
      int cpu_idx = std::strtol(basename.substr(3).c_str(), nullptr, 0);
      cpu_idx_set.emplace(cpu_idx);
    }
  }
  if (cpu_idx_set.size() > 0) {
    CPU_ZERO(cpuset);
    for (auto& idx : cpu_idx_set) {
      CPU_SET(idx, cpuset);
    }
  }
  return;
default_cpuset:
  return;
}

void ParseCoreListEntry(const std::string& s, std::unordered_set<int>& cpuset) {
  bool is_range = false;
  int acc = 0;
  int start = 0;
  for (int i = 0; i < s.size(); i++) {
    char c = s[i];
    if (c == '-') {
      is_range = true;
      start = acc;
      acc = 0;
    } else {
      if (!std::isdigit(c)) LOG(INFO) << "not digit: " << c;
      CHECK(std::isdigit(c));
      acc = acc * 10 + (c - '0');
    }
  }
  if (is_range) {
    int cpu_end = acc;
    if (s[s.size() - 1] == '-') cpu_end = sysconf(_SC_NPROCESSORS_CONF) - 1;
    for (int i = start; i <= cpu_end; i++) {
      cpuset.emplace(i);
    }
  } else {
    cpuset.emplace(acc);
  }
}

void SetAddressPort(union SocketAddress* addr, uint16_t port) {
  int af = addr->sa.sa_family;
  CHECK(af == AF_INET || af == AF_INET6);
  if (af == AF_INET) {
    addr->sin.sin_port = htons(port);
  } else {
    addr->sin6.sin6_port = htons(port);
  }
}

uint16_t GetAddressPort(const union SocketAddress* addr) {
  int af = addr->sa.sa_family;
  CHECK(af == AF_INET || af == AF_INET6);
  if (af == AF_INET) {
    return ntohs(addr->sin.sin_port);
  } else {
    return ntohs(addr->sin6.sin6_port);
  }
}

union SocketAddress AddressFromStr(const std::string& str) {
  union SocketAddress addr;

  bool is_ipv6 = false;
  if (!absl::StrContains(str, '.')) {
    is_ipv6 = true;
  }
  int sa_family = is_ipv6 ? AF_INET6 : AF_INET;
  void* dst = is_ipv6 ? (void*)&addr.sin6.sin6_addr : (void*)&addr.sin.sin_addr;
  int ret = inet_pton(sa_family, str.c_str(), dst);
  CHECK(ret == 1);
  addr.sa.sa_family = sa_family;
  return addr;
}

std::string AddressToStr(const union SocketAddress* addr) {
  char buf[256];
  const char* addr_ptr = nullptr;
  if (addr->sa.sa_family == AF_INET) {
    addr_ptr = (const char*)&addr->sin.sin_addr;
  } else {
    addr_ptr = (const char*)&addr->sin6.sin6_addr;
  }
  const char* s = inet_ntop(addr->sa.sa_family, addr_ptr, buf, 256);
  return std::string(s);
}

void SetPortnum(union SocketAddress* addr, uint16_t port) {
  if (addr->sa.sa_family == AF_INET) {
    addr->sin.sin_port = htons(port);
  } else {
    addr->sin6.sin6_port = htons(port);
  }
}

uint16_t GetPortnum(const union SocketAddress* addr) {
  if (addr->sa.sa_family == AF_INET) {
    return ntohs(addr->sin.sin_port);
  } else {
    return ntohs(addr->sin6.sin6_port);
  }
}

void BindAndListen(int fd, union SocketAddress* addr, int backlog) {
  int af = addr->sa.sa_family;
  if (af == AF_INET) {
    BindAndListen(fd, &addr->sin, backlog);
  } else {
    BindAndListen(fd, &addr->sin6, backlog);
  }
}

void BindAndListen(int fd, struct sockaddr* addr, socklen_t addrlen,
                   int backlog) {
  int ret;
  ret = bind(fd, addr, addrlen);
  PCHECK(ret == 0);

  ret = listen(fd, backlog);
  PCHECK(ret == 0);
}

void BindAndListen(int fd, struct sockaddr_in* bind_addr, int backlog) {
  BindAndListen(fd, (struct sockaddr*)bind_addr, sizeof(struct sockaddr_in),
                backlog);
}

void BindAndListen(int fd, struct sockaddr_in6* bind_addr, int backlog) {
  BindAndListen(fd, (struct sockaddr*)bind_addr, sizeof(struct sockaddr_in6),
                backlog);
}

void ConnectWithRetry(int fd, union SocketAddress* addr, int max_retry) {
  int af = addr->sa.sa_family;
  if (af == AF_INET) {
    ConnectWithRetry(fd, &addr->sin, max_retry);
  } else {
    ConnectWithRetry(fd, &addr->sin6, max_retry);
  }
}

void ConnectWithRetry(int fd, struct sockaddr* addr, socklen_t addrlen,
                      int max_retry) {
  int num_retry = 0;
  while (connect(fd, addr, addrlen) < 0) {
    LOG(INFO) << "Connection Failed";
    sleep(1);
    num_retry++;
    if (num_retry > max_retry) {
      LOG(FATAL) << "Connection Failed after " << max_retry << " retries.";
    }
  }
}

void ConnectWithRetry(int fd, struct sockaddr_in* addr, int max_retry) {
  ConnectWithRetry(fd, (struct sockaddr*)addr, sizeof(struct sockaddr_in),
                   max_retry);
}

void ConnectWithRetry(int fd, struct sockaddr_in6* addr, int max_retry) {
  ConnectWithRetry(fd, (struct sockaddr*)addr, sizeof(struct sockaddr_in6),
                   max_retry);
}

void ConnectWithRetry(int fd, const union SocketAddress* addr,
                      const union SocketAddress* local_addr, int max_retry) {
  if (local_addr) {
    socklen_t addr_len = (local_addr->sa.sa_family == AF_INET)
                             ? sizeof(struct sockaddr_in)
                             : sizeof(struct sockaddr_in6);
    SetReuseAddr(fd);
    if (bind(fd, &local_addr->sa, addr_len) < 0) {
      LOG(FATAL) << absl::StrFormat("bind failed: %s", strerror(errno));
    }
  }
  socklen_t addr_len = (addr->sa.sa_family == AF_INET)
                           ? sizeof(struct sockaddr_in)
                           : sizeof(struct sockaddr_in6);
  int num_retry = 0;
  while (connect(fd, &addr->sa, addr_len) < 0) {
    LOG(INFO) << "Connection Attempt" << num_retry
              << "Failed: " << strerror(errno);
    sleep(1);
    num_retry++;
    if (num_retry > max_retry) {
      LOG(FATAL) << "Connection Failed after " << max_retry << " retries.";
    }
  }
}

void SendMyAddresses(const std::vector<union SocketAddress>& my_addresses,
                     int fd) {
  size_t msg_sz = my_addresses.size() * sizeof(union SocketAddress);
  ssize_t ret = send(fd, &msg_sz, sizeof(msg_sz), 0);
  PCHECK(ret == sizeof(msg_sz));
  size_t bytes_sent = 0;
  char* ptr = (char*)my_addresses.data();
  while (bytes_sent < msg_sz) {
    ssize_t ret = send(fd, &ptr[bytes_sent], msg_sz - bytes_sent, 0);
    PCHECK(ret >= 0);
    bytes_sent += ret;
  }
}

void ReceivePeerAddresses(std::vector<union SocketAddress>* peer_addresses,
                          int fd) {
  size_t msg_sz = 0;
  ssize_t ret = recv(fd, &msg_sz, sizeof(msg_sz), 0);
  PCHECK(ret == sizeof(msg_sz));
  CHECK(msg_sz % sizeof(union SocketAddress) == 0);
  peer_addresses->resize(msg_sz / sizeof(union SocketAddress));

  size_t bytes_recv = 0;
  char* ptr = (char*)peer_addresses->data();
  while (bytes_recv < msg_sz) {
    ssize_t ret = recv(fd, &ptr[bytes_recv], msg_sz - bytes_recv, 0);
    PCHECK(ret >= 0);
    bytes_recv += ret;
  }
}

void ServerAcceptControlChannelConnection(
    union SocketAddress* server_addr,
    std::vector<union SocketAddress>* client_hpn_addresses,
    const std::vector<union SocketAddress>& server_hpn_addresses) {
  int listen_fd = CreateTcpSocket(server_addr->sa.sa_family);
  SetReuseAddr(listen_fd);
  BindAndListen(listen_fd, server_addr, 1);

  union SocketAddress client_addr;
  socklen_t client_addr_len = sizeof(client_addr);
  int fd = accept(listen_fd, &client_addr.sa, &client_addr_len);

  SendMyAddresses(server_hpn_addresses, fd);
  ReceivePeerAddresses(client_hpn_addresses, fd);

  close(fd);
  close(listen_fd);
}

void ClientConnectControlChannel(
    union SocketAddress* server_addr,
    std::vector<union SocketAddress>* server_hpn_addresses,
    const std::vector<union SocketAddress>& client_hpn_addresses) {
  int fd = CreateTcpSocket(server_addr->sa.sa_family);
  ConnectWithRetry(fd, server_addr);

  ReceivePeerAddresses(server_hpn_addresses, fd);
  SendMyAddresses(client_hpn_addresses, fd);

  close(fd);
}

int ReadNicPci(const char* ifname, uint16_t* dbdf) {
  static const char* k_match_prefix = "PCI_SLOT_NAME=";
  int match_prefix_len = strlen(k_match_prefix);
  int uevent_fd;
  ssize_t ret;
  ssize_t off = 0;
  bool found_matching_line = false;
  int i;
  char sysfs_path[PATH_MAX];
  char line[256];
  sprintf(sysfs_path, "/sys/class/net/%s/device/uevent", ifname);
  uevent_fd = open(sysfs_path, O_RDONLY);
  if (uevent_fd < 0) {
    return -1;
  }

  do {
    ssize_t last_newline = -1;
    ssize_t line_start_off = 0;
    ret = read(uevent_fd, &line[off], 256 - off);
    if (ret < 0) {
      fprintf(stderr, "error while reading: ");
      close(uevent_fd);
      return -1;
    }

    for (i = 0; i < ret; i++) {
      if (line[off + i] == '\n') {
        // line[0:off+i] is a whole line
        line[off + i] = '\0';
        if (off + i - line_start_off > match_prefix_len &&
            strncmp(k_match_prefix, &line[line_start_off], match_prefix_len) ==
                0) {
          sscanf(&line[line_start_off + match_prefix_len], "%hx:%hx:%hx.%hx",
                 &dbdf[0], &dbdf[1], &dbdf[2], &dbdf[3]);
          found_matching_line = true;
          break;
        }
        last_newline = off + i;
        line_start_off = off + i + 1;
      }
    }
    if (found_matching_line) break;
    if (last_newline >= 0) {
      for (i = last_newline + 1; i < off + ret; i++) {
        line[i - last_newline - 1] = line[i];
      }
      off = off + ret - last_newline;
    } else {
      off += ret;
      if (off >= 256) {
        fprintf(stderr, "line longer than 256 bytes");
        break;
      }
    }
  } while (ret > 0);

  close(uevent_fd);
  if (found_matching_line)
    return 0;
  else
    return -1;
}

void DiscoverNetif(std::vector<NetifInfo>& nic_info) {
  auto ifname_filter = [](struct ifaddrs* interface) -> bool {
    int af = interface->ifa_addr->sa_family;
    if (af != AF_INET && af != AF_INET6) {
      return false;
    }
    std::string ifname(interface->ifa_name);
    for (auto& e : kIgnoredIf) {
      if (e == ifname) {
        return false;
      }
    }
    for (auto& e : kIgnoredIfPrefix) {
      if (absl::StartsWith(ifname, e)) {
        return false;
      }
    }
    if (af == AF_INET6) {
      char buf[256];
      struct sockaddr_in6* addr = (struct sockaddr_in6*)interface->ifa_addr;
      const char* s = inet_ntop(af, &addr->sin6_addr, buf, 256);
      std::string addr_str(s);
      // LOG(LOG_INFO, "scanning if: %s %s\n", ifname.c_str(),
      // addr_str.c_str());
      if (absl::StartsWith(addr_str, kIpv6LinkLocalPrefix)) {
        return false;
      }
    }
    return true;
  };
  DiscoverNetif(nic_info, ifname_filter);
}
}  // namespace gpudirect_tcpxd
