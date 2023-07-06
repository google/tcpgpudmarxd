#include "include/rx_rule_client.h"

#include <absl/functional/bind_front.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <arpa/inet.h>
#include <cstring>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iterator>
#include <sys/un.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "include/flow_steer_ntuple.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_manager.h"
#include "include/socket_helper.h"
#include "proto/unix_socket_message.pb.h"
#include "gmock/gmock.h"
#include <google/protobuf/text_format.h>

namespace {
using tcpdirect::SocketAddress;

TEST(SocketHelperTest, SetAddressPortIpv4) {
  SocketAddress socket_address;
  socket_address.sa.sa_family = AF_INET;
  tcpdirect::SetAddressPort(&socket_address, 1234);
  EXPECT_EQ(socket_address.sin.sin_port, htons(1234));
}

TEST(SocketHelperTest, SetAddressPortIpv6) {
  SocketAddress socket_address;
  socket_address.sa.sa_family = AF_INET6;
  tcpdirect::SetAddressPort(&socket_address, 1234);
  EXPECT_EQ(socket_address.sin6.sin6_port, htons(1234));
}

TEST(SocketHelperTest, GetAddressPortIpv4) {
  SocketAddress socket_address;
  socket_address.sa.sa_family = AF_INET;
  socket_address.sin.sin_port = htons(1234);
  EXPECT_EQ(socket_address.sin.sin_port, htons(1234));
}

TEST(SocketHelperTest, GetAddressPortIpv6) {
  SocketAddress socket_address;
  socket_address.sa.sa_family = AF_INET;
  socket_address.sin6.sin6_port = htons(1234);
  EXPECT_EQ(socket_address.sin6.sin6_port, htons(1234));
}

TEST(SocketHelperTest, AddressFromStrIpv4) {
  std::string address_str = "1.2.3.4";
  SocketAddress socket_address;
  int sa_family = AF_INET;
  void *dst = (void *)&socket_address.sin.sin_addr;
  int ret = inet_pton(sa_family, address_str.c_str(), dst);

  auto addr = tcpdirect::AddressFromStr(address_str);
  EXPECT_EQ(memcmp(&addr.sin.sin_addr, dst, sizeof(struct in_addr )), 0);
}

TEST(SocketHelperTest, AddressFromStrIpv6) {
  std::string address_str = "2001:db8::1:0";
  SocketAddress socket_address;
  int sa_family = AF_INET6;
  void *dst = (void *)&socket_address.sin6.sin6_addr;
  int ret = inet_pton(sa_family, address_str.c_str(), dst);

  auto addr = tcpdirect::AddressFromStr(address_str);
  EXPECT_EQ(memcmp(&addr.sin6.sin6_addr, dst, sizeof(struct in_addr )), 0);
}


TEST(SocketHelperTest, AddressToStrIpv4){
  std::string addr_str_1 = "1.2.3.4";
  SocketAddress socket_address;
  int sa_family = AF_INET;
  void *dst = (void *)&socket_address.sin.sin_addr;
  int ret = inet_pton(sa_family, addr_str_1.c_str(), dst);
  socket_address.sa.sa_family = sa_family;

  auto addr_str_2 = tcpdirect::AddressToStr(&socket_address);
  EXPECT_EQ(addr_str_2, addr_str_1);
}

TEST(SocketHelperTest, AddressToStrIpv6) {
  std::string addr_str_1 = "2001:db8::1:0";
  SocketAddress socket_address;
  int sa_family = AF_INET6;
  void *dst = (void *)&socket_address.sin6.sin6_addr;
  int ret = inet_pton(sa_family, addr_str_1.c_str(), dst);
  socket_address.sa.sa_family = sa_family;

  auto addr_str_2 = tcpdirect::AddressToStr(&socket_address);
  EXPECT_EQ(addr_str_2, addr_str_1);
}
} // namespace