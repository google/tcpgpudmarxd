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

#include <absl/functional/bind_front.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <arpa/inet.h>
#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include <sys/un.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "include/flow_steer_ntuple.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_client.h"
#include "include/rx_rule_manager.h"
#include "proto/unix_socket_message.pb.h"

namespace {
using gpudirect_tcpxd::SocketAddress;

TEST(SocketHelperTest, SetAddressPortIpv4) {
  SocketAddress socket_address;
  socket_address.sa.sa_family = AF_INET;
  gpudirect_tcpxd::SetAddressPort(&socket_address, 1234);
  EXPECT_EQ(socket_address.sin.sin_port, htons(1234));
}

TEST(SocketHelperTest, SetAddressPortIpv6) {
  SocketAddress socket_address;
  socket_address.sa.sa_family = AF_INET6;
  gpudirect_tcpxd::SetAddressPort(&socket_address, 1234);
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

  auto addr = gpudirect_tcpxd::AddressFromStr(address_str);
  EXPECT_EQ(memcmp(&addr.sin.sin_addr, dst, sizeof(struct in_addr)), 0);
}

TEST(SocketHelperTest, AddressFromStrIpv6) {
  std::string address_str = "2001:db8::1:0";
  SocketAddress socket_address;
  int sa_family = AF_INET6;
  void *dst = (void *)&socket_address.sin6.sin6_addr;
  int ret = inet_pton(sa_family, address_str.c_str(), dst);

  auto addr = gpudirect_tcpxd::AddressFromStr(address_str);
  EXPECT_EQ(memcmp(&addr.sin6.sin6_addr, dst, sizeof(struct in_addr)), 0);
}

TEST(SocketHelperTest, AddressToStrIpv4) {
  std::string addr_str_1 = "1.2.3.4";
  SocketAddress socket_address;
  int sa_family = AF_INET;
  void *dst = (void *)&socket_address.sin.sin_addr;
  int ret = inet_pton(sa_family, addr_str_1.c_str(), dst);
  socket_address.sa.sa_family = sa_family;

  auto addr_str_2 = gpudirect_tcpxd::AddressToStr(&socket_address);
  EXPECT_EQ(addr_str_2, addr_str_1);
}

TEST(SocketHelperTest, AddressToStrIpv6) {
  std::string addr_str_1 = "2001:db8::1:0";
  SocketAddress socket_address;
  int sa_family = AF_INET6;
  void *dst = (void *)&socket_address.sin6.sin6_addr;
  int ret = inet_pton(sa_family, addr_str_1.c_str(), dst);
  socket_address.sa.sa_family = sa_family;

  auto addr_str_2 = gpudirect_tcpxd::AddressToStr(&socket_address);
  EXPECT_EQ(addr_str_2, addr_str_1);
}
}  // namespace