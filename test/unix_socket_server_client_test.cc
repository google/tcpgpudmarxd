// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <errno.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "include/unix_socket_client.h"
#include "include/unix_socket_server.h"
#include "proto/unix_socket_message.pb.h"

namespace {

using gpudirect_tcpxd::UnixSocketClient;
using gpudirect_tcpxd::UnixSocketMessage;
using gpudirect_tcpxd::UnixSocketServer;

#define EXPECT_OK(status) EXPECT_TRUE(status.ok())

TEST(UnixSocketServerTest, EchoServer) {
  std::string socket_path = absl::StrFormat("/tmp/echo_server");
  const char* hello_from_client = "hello from client\n";

  pid_t pid = fork();

  if (pid == 0) {
    // child
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    UnixSocketClient client(socket_path);
    EXPECT_OK(client.Connect());
    // basic echo
    UnixSocketMessage req;
    req.mutable_proto()->set_raw_bytes("test");
    client.Send(req);
    absl::StatusOr<UnixSocketMessage> resp = client.Receive();
    EXPECT_OK(resp.status());
    EXPECT_TRUE(resp.value().has_proto());
    EXPECT_EQ(resp.value().proto().raw_bytes(), "test");
    // fd transfer
    UnixSocketMessage reqfd;
    reqfd.mutable_proto()->set_raw_bytes("givemefd");
    client.Send(reqfd);
    absl::StatusOr<UnixSocketMessage> resp_fd = client.Receive();
    EXPECT_OK(resp_fd.status());
    EXPECT_TRUE(resp_fd.value().has_fd());
    int fd = resp_fd.value().fd();
    EXPECT_EQ(write(fd, hello_from_client, strlen(hello_from_client)),
              strlen(hello_from_client));
    close(fd);
    // basic echo again
    UnixSocketMessage req2;
    req2.mutable_proto()->set_raw_bytes("test");
    client.Send(req2);
    absl::StatusOr<UnixSocketMessage> resp2 = client.Receive();
    EXPECT_OK(resp2.status());
    EXPECT_TRUE(resp2.value().has_proto());
    EXPECT_EQ(resp2.value().proto().raw_bytes(), "test");
    // close down
    UnixSocketMessage req_down;
    req_down.mutable_proto()->set_raw_bytes("down");
    client.Send(req_down);
    exit(0);
  } else {
    // parent
    int pipefd[2];
    EXPECT_EQ(pipe(pipefd), 0);

    UnixSocketServer server(
        socket_path, [pipefd](UnixSocketMessage&& request,
                              UnixSocketMessage* response, bool* fin) {
          if (request.has_proto() && request.proto().raw_bytes() == "down") {
            *fin = true;
          } else if (request.has_proto() &&
                     request.proto().raw_bytes() == "givemefd") {
            response->set_fd(pipefd[1]);
          } else {
            *response = std::move(request);
          }
        });

    EXPECT_OK(server.Start());
    std::this_thread::sleep_for(std::chrono::milliseconds{1000});
    server.Stop();
    // Check fd
    char buf[100];
    close(pipefd[1]);
    EXPECT_GT(read(pipefd[0], buf, 100), 0);
    EXPECT_EQ(strncmp(buf, hello_from_client, strlen(hello_from_client)), 0);
    close(pipefd[0]);
    wait(nullptr);
  }
}

}  // namespace
