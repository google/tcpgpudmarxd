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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "include/application_registry_client.h"
#include "include/application_registry_manager.h"

namespace {
using gpudirect_tcpxd::ApplicationRegistryClient;
using gpudirect_tcpxd::ApplicationRegistryManager;

static std::atomic<bool> gShouldStop(false);

void sig_handler(int signum) {
  if (signum == SIGTERM) {
    gShouldStop.store(true, std::memory_order_release);
  }
}

// Cleanup when there is one client.
TEST(ApplicationRegistryTest, CleanupSucceed) {
  gShouldStop.store(false);
  signal(SIGTERM, sig_handler);
  std::unique_ptr<ApplicationRegistryManager> application_registry_manager;
  pthread_t main_id = pthread_self();
  application_registry_manager = std::make_unique<ApplicationRegistryManager>(
      /*prefix=*/"/tmp", main_id);
  auto manager_status = application_registry_manager->Init();
  EXPECT_TRUE(manager_status.ok());
  std::unique_ptr<ApplicationRegistryClient> application_registry_client;
  application_registry_client =
      std::make_unique<ApplicationRegistryClient>("/tmp");
  auto client_status = application_registry_client->Init();
  EXPECT_TRUE(client_status.ok());
  application_registry_client.reset();
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_TRUE(gShouldStop.load());
}

// Cleanup when there are multiple clients.
TEST(ApplicationRegistryTest, MultipleClientsCleanupSucceed) {
  gShouldStop.store(false);
  signal(SIGTERM, sig_handler);
  std::unique_ptr<ApplicationRegistryManager> application_registry_manager;
  pthread_t main_id = pthread_self();
  application_registry_manager = std::make_unique<ApplicationRegistryManager>(
      /*prefix=*/"/tmp", main_id);
  auto manager_status = application_registry_manager->Init();
  EXPECT_TRUE(manager_status.ok());
  std::vector<std::unique_ptr<ApplicationRegistryClient>> clients;
  for (int i = 0; i < 3; ++i) {
    auto application_registry_client =
        std::make_unique<ApplicationRegistryClient>("/tmp");
    auto client_status = application_registry_client->Init();
    EXPECT_TRUE(client_status.ok());
    clients.push_back(std::move(application_registry_client));
  }
  EXPECT_FALSE(gShouldStop.load());
  for (int i = 0; i < clients.size(); ++i) {
    clients[i].reset();
    absl::SleepFor(absl::Milliseconds(100));
    if (i < clients.size() - 1) {
      EXPECT_FALSE(gShouldStop.load());
    } else {
      EXPECT_TRUE(gShouldStop.load());
    }
  }
}
}  // namespace