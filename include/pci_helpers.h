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

#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PCI_HELPERS_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PCI_HELPERS_H_

#include <stdio.h>
#include <sys/types.h>

#include <string>
#include <vector>

namespace tcpdirect {

// Get PCI address of a netwok if
int read_nic_pci_addr(const char* ifname, uint16_t* domain, uint16_t* bus,
                      uint16_t* device);

// List all child devices starting from a PCI device that has the
// specified vendor ID

int list_vendor_devices(const char* parent_dir_path,
                        std::vector<std::string>* candidates,
                        const char* vendor_id);

int parse_pci_addr(const char* pci_addr, uint16_t* domain, uint16_t* bus,
                   uint16_t* device, uint16_t* function);
}  // namespace tcpdirect

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PCI_HELPERS_H_
