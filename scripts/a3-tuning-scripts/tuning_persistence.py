"""A package that applies the tcp tuning if necessary every 5s.
"""

import datetime
import os
import subprocess
import time
import sys

cmd = "ip route"
check_intervals = 5
initcwnd = str(sys.argv[1])
rto_min = str(sys.argv[2])+"ms"
quickack = str(sys.argv[3])
def AddTuningIfNeeded(prefix_cmd: str, ip_route: str):
  """Add a3 tcp tuning to an ip route if needed.

  Args:
    prefix_cmd: prefix added to each command to specify netns
    ip_route: ip route to be potentially modified
  """
  if "initcwnd" in ip_route and "rto_min" in ip_route:
    return

  ip_route = ip_route.replace("rto_min lock", "rto_min")
  # Skipping linkdown routes, they cannot be deleted or tuned
  if "linkdown" in ip_route:
    return
  ip_route_modified = ip_route;
  if (initcwnd != "0" ):
    ip_route_modified += " initcwnd " + initcwnd
  if (rto_min != "0ms" ):
    ip_route_modified += " rto_min " + rto_min
  if (quickack != "0" ):
     ip_route_modified += " quickack " + quickack

  os.system(prefix_cmd + "ip route replace " + ip_route_modified)
  print(datetime.datetime.now(), " Adding tuning to route " + ip_route)


def FixTcpTunings(netns_domains: list[str]):
  """Reapply tcp tuning to all ip routes in the system if necessary.

  Args:
    netns_domains: list of netns domains that the tuning would be applied to
  """
  for netns_domain in netns_domains:
    prefix_cmd = ""
    if netns_domain:
      prefix_cmd = "ip netns exec " + netns_domain
    ip_routes_results = subprocess.run(
        prefix_cmd + cmd,
        capture_output=True,
        shell=True,
        text=True,
        check=False,
    )
    ip_routes = ip_routes_results.stdout.splitlines(keepends=False)
    for ip_route in ip_routes:
      AddTuningIfNeeded(prefix_cmd, ip_route)
    print(
        datetime.datetime.now(),
        " tcp_tuner finish fixing tcp tuning for netns domain " + netns_domain,
    )


def FindAllNetnsDomains() -> list[str]:
  """Find all netns domains in the host.
  """
  get_domains_command = "ip netns list"
  netns_results = subprocess.run(
      get_domains_command,
      capture_output=True,
      shell=True,
      text=True,
      check=False,
  )
  netns_results = netns_results.stdout.splitlines(keepends=False)
  netns_domains = [""]
  for netns_domain in netns_results:
    netns_domains.append(netns_domain.split()[0])
  return netns_domains



print("initcwnd is " + initcwnd)
print("rto_mins is " + rto_min)
print("quickack is " + quickack)
while (True):
  netns_domain_names = FindAllNetnsDomains()
  FixTcpTunings(netns_domain_names)
  time.sleep(check_intervals)

