
#include <linux/ethtool.h>
#include <linux/in.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include "third_party/absl/flags/flag.h"
#include "third_party/absl/flags/parse.h"
#include "third_party/absl/strings/substitute.h"

ABSL_FLAG(std::string, interface, "", "interface name");

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  std::string interface = absl::GetFlag(FLAGS_interface);
  if (interface.empty()) {
    std::cout << "Need to specify interface" << std::endl;
    return -1;
  }

  auto fd = socket(AF_INET6, SOCK_DGRAM, IPPROTO_IP);
  if (fd < 0) {
    std::cout << "Unable to open socket" << std::endl;
    return -1;
  }

  struct ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));
  strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ);

  struct ethtool_drvinfo drvinfo = {};
  drvinfo.cmd = ETHTOOL_GDRVINFO;
  ifr.ifr_data = reinterpret_cast<char *>(&drvinfo);
  int error = ioctl(fd, SIOCETHTOOL, &ifr);
  if (error < 0) {
    std::cout << absl::Substitute("ioctl() SIOCETHTOOL failed for device $0",
                                  interface)
              << std::endl;
    return -1;
  }
  std::cout << "Interface " << interface << " has driver name "
            << drvinfo.driver << std::endl;
  close(fd);

  return 0;
}