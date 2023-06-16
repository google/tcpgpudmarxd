#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_SOCKET_HELPER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_SOCKET_HELPER_H_

#include <netinet/in.h>

#include <string>
#include <vector>

namespace tcpdirect {

union SocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
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
}  // namespace tcpdirect

#endif /* _THIRD_PARTY_TCPDIRECT_RX_MANAGER_BENCHMARK_SOCKET_HELPER_H_ */
