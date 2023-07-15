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

#include "include/pci_helpers.h"

#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <linux/limits.h>
#include <stdio.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include <absl/log/log.h>

namespace tcpdirect {

#define PCI_VENDOR_LEN 6

int parse_pci_addr(const char* pci_addr, uint16_t* domain, uint16_t* bus,
                   uint16_t* device, uint16_t* function) {
  uint16_t tmp_domain, tmp_bus, tmp_device, tmp_function;
  if (sscanf(pci_addr, "%hx:%hx:%hx.%hx", &tmp_domain, &tmp_bus, &tmp_device,
             &tmp_function) != 4)
    return -1;
  *domain = tmp_domain;
  *bus = tmp_bus;
  *device = tmp_device;
  *function = tmp_function;
  return 0;
}


int read_nic_pci_addr(const char* ifname, uint16_t* domain, uint16_t* bus,
                      uint16_t* device) {
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
    LOG(ERROR) << "failed to open file: " << sysfs_path;
    return -1;
  }

  do {
    ssize_t last_newline = -1;
    ssize_t line_start_off = 0;
    ret = read(uevent_fd, &line[off], 256 - off);
    if (ret < 0) {
      LOG(ERROR) << "error while reading: " << strerror(errno);
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
          sscanf(&line[line_start_off + match_prefix_len], "%hx:%hx:%hx.0",
                 domain, bus, device);
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
        LOG(ERROR) << "line longer than 256 bytes";
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

int list_vendor_devices(const char* parent_dir_path,
                        std::vector<std::string>* candidates,
                        const char* vendor_id) {
  DIR *root_dir = opendir(parent_dir_path);
  if (root_dir == nullptr) {
    LOG(ERROR) << "Failed to open parent directory: " << parent_dir_path;
    return -1;
  }
  struct dirent* dir_entry;
  char subdir_path[PATH_MAX];
  char vendor[PCI_VENDOR_LEN + 1] = { 0 };
  uint16_t domain, bus, device, function;
  while ((dir_entry = readdir(root_dir)) != nullptr) {
    if (parse_pci_addr(dir_entry->d_name, &domain, &bus, &device, &function))
      continue;
    snprintf(subdir_path, PATH_MAX, "%s/%s/vendor", parent_dir_path,
             dir_entry->d_name);
    int fd = open(subdir_path, O_RDONLY);
    if (fd < 0) continue;
    if (read(fd, vendor, PCI_VENDOR_LEN) < PCI_VENDOR_LEN) {
      LOG(ERROR) << "Invalid vendor ID format";
    } else {
      // NVIDIA PCI Vendor ID
      if (!strncmp(vendor, "0x10de", PCI_VENDOR_LEN)) {
        LOG(INFO) << "Match: PCI address " << dir_entry->d_name;
        candidates->emplace_back(dir_entry->d_name);
      }
    }
    close(fd);
    snprintf(subdir_path, PATH_MAX, "%s/%s", parent_dir_path,
             dir_entry->d_name);
    // This is not a leaf node, continue our search
    list_vendor_devices(subdir_path, candidates, vendor_id);
  }
  return 0;
}

}  // namespace tcpdirect