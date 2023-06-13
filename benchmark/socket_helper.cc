#include "experimental/users/chechenglin/tcpgpudmad/benchmark/socket_helper.h"

#include <arpa/inet.h>

#include <string>
#include <vector>

#include "base/logging.h"
#include "experimental/users/chechenglin/tcpgpudmad/benchmark/benchmark_common.h"
#include "third_party/absl/strings/match.h"

namespace tcpdirect {
int CreateTcpSocket(int address_family) {
  int fd = socket(address_family, SOCK_STREAM, 0);
  PCHECK(fd >= 0);
  return fd;
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

void SetAddressPort(union SocketAddress* addr, uint16_t port) {
  int af = addr->sa.sa_family;
  CHECK(af == AF_INET || af == AF_INET6);
  if (af == AF_INET) {
    addr->sin.sin_port = htons(port);
  } else {
    addr->sin6.sin6_port = htons(port);
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

void AcceptConnectionAndSendHpnAddress(
    union SocketAddress* server_addr,
    const std::vector<union SocketAddress>& server_hpn_addresses) {
  int listen_fd = CreateTcpSocket(server_addr->sa.sa_family);
  SetReuseAddr(listen_fd);
  BindAndListen(listen_fd, server_addr, 1);

  union SocketAddress client_addr;
  socklen_t client_addr_len = sizeof(client_addr);
  int fd = accept(listen_fd, &client_addr.sa, &client_addr_len);

  size_t msg_sz = server_hpn_addresses.size() * sizeof(union SocketAddress);
  ssize_t ret = send(fd, &msg_sz, sizeof(msg_sz), 0);
  PCHECK(ret == sizeof(msg_sz));
  size_t bytes_sent = 0;
  char* ptr = (char*)server_hpn_addresses.data();
  while (bytes_sent < msg_sz) {
    ssize_t ret = send(fd, &ptr[bytes_sent], msg_sz - bytes_sent, 0);
    PCHECK(ret >= 0);
    bytes_sent += ret;
  }
  close(fd);
  close(listen_fd);
}

void ConnectAndReceiveHpnAddress(
    union SocketAddress* server_addr,
    std::vector<union SocketAddress>* server_hpn_addresses) {
  int fd = CreateTcpSocket(server_addr->sa.sa_family);
  ConnectWithRetry(fd, server_addr);

  size_t msg_sz = 0;
  ssize_t ret = recv(fd, &msg_sz, sizeof(msg_sz), 0);
  PCHECK(ret == sizeof(msg_sz));
  CHECK(msg_sz % sizeof(union SocketAddress) == 0);
  server_hpn_addresses->resize(msg_sz / sizeof(union SocketAddress));

  size_t bytes_recv = 0;
  char* ptr = (char*)server_hpn_addresses->data();
  while (bytes_recv < msg_sz) {
    ssize_t ret = recv(fd, &ptr[bytes_recv], msg_sz - bytes_recv, 0);
    PCHECK(ret >= 0);
    bytes_recv += ret;
  }
  close(fd);
}
}  // namespace tcpdirect
