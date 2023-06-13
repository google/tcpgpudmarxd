#include "include/ioctl_nic_configurator.h"

#include <linux/ethtool.h>
#include <linux/netlink.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include "include/ethtool_common.h"
#include <absl/status/status.h>
#include <absl/strings/str_format.h>

#ifndef SIOCETHTOOL
#define SIOCETHTOOL 0x8946
#endif

namespace tcpdirect {

struct ethtool_gstrings *IoctlNicConfigurator::GetStringsSet(
    const std::string &ifname, ethtool_stringset set_id,
    ptrdiff_t drvinfo_offset, int null_terminate) {
  struct {
    struct ethtool_sset_info hdr;
    uint32_t buf[1];
  } sset_info;
  struct ethtool_drvinfo drvinfo;
  uint32_t len, i;
  struct ethtool_gstrings *strings;

  sset_info.hdr.cmd = ETHTOOL_GSSET_INFO;
  sset_info.hdr.reserved = 0;
  sset_info.hdr.sset_mask = 1ULL << set_id;
  if (SendIoctl(ifname, (char *)&sset_info) == 0) {
    const uint32_t *sset_lengths = sset_info.hdr.data;

    len = sset_info.hdr.sset_mask ? sset_lengths[0] : 0;
  } else if (errno == EOPNOTSUPP && drvinfo_offset != 0) {
    /* Fallback for old kernel versions */
    drvinfo.cmd = ETHTOOL_GDRVINFO;
    if (SendIoctl(ifname, (char *)&drvinfo)) return nullptr;
    len = *(uint32_t *)((char *)&drvinfo + drvinfo_offset);
  } else {
    return nullptr;
  }

  strings = (struct ethtool_gstrings *)calloc(
      1, sizeof(*strings) + len * ETH_GSTRING_LEN);
  if (!strings) return nullptr;

  strings->cmd = ETHTOOL_GSTRINGS;
  strings->string_set = set_id;
  strings->len = len;
  if (len != 0 && SendIoctl(ifname, (char *)strings)) {
    free(strings);
    return nullptr;
  }

  if (null_terminate) {
    for (i = 0; i < len; i++) strings->data[(i + 1) * ETH_GSTRING_LEN - 1] = 0;
  }

  return strings;
}

struct feature_defs *IoctlNicConfigurator::GetFeatureDefs(
    const std::string &ifname) {
  struct ethtool_gstrings *names;
  struct feature_defs *defs;
  unsigned int i, j;
  uint32_t n_features;

  names = GetStringsSet(ifname, ETH_SS_FEATURES, 0, 1);
  if (names) {
    n_features = names->len;
  } else if (errno == EOPNOTSUPP || errno == EINVAL) {
    /* Kernel doesn't support named features; not an error */
    n_features = 0;
  } else if (errno == EPERM) {
    /* Kernel bug: ETHTOOL_GSSET_INFO was privileged.
     * Work around it. */
    n_features = 0;
  } else {
    return nullptr;
  }

  defs = (struct feature_defs *)malloc(sizeof(*defs) +
                                       sizeof(defs->def[0]) * n_features);
  if (!defs) {
    free(names);
    return nullptr;
  }

  defs->n_features = n_features;
  memset(defs->off_flag_matched, 0, sizeof(defs->off_flag_matched));

  /* Copy out feature names and find those associated with legacy flags */
  for (i = 0; i < defs->n_features; i++) {
    memcpy(defs->def[i].name, names->data + i * ETH_GSTRING_LEN,
           ETH_GSTRING_LEN);
    defs->def[i].off_flag_index = -1;

    for (j = 0; j < OFF_FLAG_DEF_SIZE && defs->def[i].off_flag_index < 0; j++) {
      const char *pattern = off_flag_def[j].kernel_name;
      const char *name = defs->def[i].name;
      for (;;) {
        if (*pattern == '*') {
          /* There is only one wildcard; so
           * switch to a suffix comparison */
          size_t pattern_len = strlen(pattern + 1);
          size_t name_len = strlen(name);
          if (name_len < pattern_len) break; /* name is too short */
          name += name_len - pattern_len;
          ++pattern;
        } else if (*pattern != *name) {
          break; /* mismatch */
        } else if (*pattern == 0) {
          defs->def[i].off_flag_index = j;
          defs->off_flag_matched[j]++;
          break;
        } else {
          ++name;
          ++pattern;
        }
      }
    }
  }

  free(names);
  return defs;
}

struct feature_state *IoctlNicConfigurator::GetFeatures(
    const std::string &ifname, const struct feature_defs *defs) {
  struct feature_state *state;
  struct ethtool_value eval;
  int err, allfail = 1;
  uint32_t value;
  int i;

  state = (struct feature_state *)malloc(
      sizeof(*state) + FEATURE_BITS_TO_BLOCKS(defs->n_features) *
                           sizeof(state->features.features[0]));
  if (!state) return nullptr;

  state->off_flags = 0;

  for (i = 0; i < OFF_FLAG_DEF_SIZE; i++) {
    value = off_flag_def[i].value;
    if (!off_flag_def[i].get_cmd) continue;
    eval.cmd = off_flag_def[i].get_cmd;
    err = SendIoctl(ifname, (char *)&eval);
    if (err) {
      if (errno == EOPNOTSUPP && off_flag_def[i].get_cmd == ETHTOOL_GUFO)
        continue;

      fprintf(stderr, "Cannot get device %s settings: %m\n",
              off_flag_def[i].long_name);
    } else {
      if (eval.data) state->off_flags |= value;
      allfail = 0;
    }
  }

  eval.cmd = ETHTOOL_GFLAGS;
  err = SendIoctl(ifname, (char *)&eval);
  if (err) {
    perror("Cannot get device flags");
  } else {
    state->off_flags |= eval.data & ETH_FLAG_EXT_MASK;
    allfail = 0;
  }

  if (defs->n_features) {
    state->features.cmd = ETHTOOL_GFEATURES;
    state->features.size = FEATURE_BITS_TO_BLOCKS(defs->n_features);
    err = SendIoctl(ifname, (char *)&state->features);
    if (err)
      perror("Cannot get device generic features");
    else
      allfail = 0;
  }

  if (allfail) {
    free(state);
    return nullptr;
  }

  return state;
}

absl::Status IoctlNicConfigurator::Init() {
  fd_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_ < 0) {
    fd_ = socket(AF_NETLINK, SOCK_RAW, NETLINK_GENERIC);
  }
  if (fd_ < 0) {
    perror("Cannot get control socket");
    return absl::ErrnoToStatus(
        errno,
        absl::StrFormat(
            "Unable to create AF_INET/NETLINK socket for ethtool, err: %d",
            errno));
  }
  return absl::OkStatus();
}

void IoctlNicConfigurator::Cleanup() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

int IoctlNicConfigurator::SendIoctl(const std::string &ifname, char *cmd) {
  struct ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));
  strncpy(ifr.ifr_name, ifname.c_str(), IFNAMSIZ);
  ifr.ifr_data = cmd;
  return ioctl(fd_, SIOCETHTOOL, &ifr);
}

absl::Status IoctlNicConfigurator::ToggleHeaderSplit(const std::string &ifname,
                                                     bool enable) {
  const char kHeaderSplitFlag[] = "enable-header-split";
  struct ethtool_gstrings *strings =
      GetStringsSet(ifname, ETH_SS_PRIV_FLAGS,
                    offsetof(struct ethtool_drvinfo, n_priv_flags), 1);
  if (!strings) {
    return absl::UnavailableError("Cannot get private flag names");
  }
  if (strings->len == 0) {
    free(strings);
    return absl::UnavailableError("No private flags defined");
  }
  if (strings->len > 32) {
    /* ETHTOOL_{G,S}PFLAGS can only cover 32 flags */
    std::cerr << "Only setting first 32 private flags" << std::endl;
    strings->len = 32;
  }

  uint32_t enabled_flag = 0, disabled_flag = 0;
  for (int i = 0; i < strings->len; i++) {
    if (!strcmp(((const char *)strings->data + i * ETH_GSTRING_LEN),
                kHeaderSplitFlag)) {
      if (enable) {
        enabled_flag |= 1U << i;
      } else {
        disabled_flag |= 1U << i;
      }
    }
  }

  struct ethtool_value flags;
  flags.cmd = ETHTOOL_GPFLAGS;
  if (SendIoctl(ifname, (char *)&flags)) {
    free(strings);
    return absl::UnavailableError("Cannot get private flags");
  }

  flags.cmd = ETHTOOL_SPFLAGS;
  flags.data = (flags.data & ~disabled_flag) | enabled_flag;
  if (SendIoctl(ifname, (char *)&flags)) {
    free(strings);
    return absl::UnavailableError("Cannot set private flags");
  }

  free(strings);
  return absl::OkStatus();
}

absl::Status IoctlNicConfigurator::SetRss(const std::string &ifname,
                                          int num_queues) {
  // The following code is the excerpt from "ethtool.c" that attempts to
  // achieve: ethtool --rxfh equal <num_queues>

  if (num_queues < 1)
    return absl::InvalidArgumentError(
        "SetRss(num_queues), num_queues should be greater than 0.");

  int err = 0;
  struct ethtool_rxfh rss_head = {0};
  struct ethtool_rxfh *rss = nullptr;
  uint32_t entry_size = sizeof(rss_head.rss_config[0]);
  uint32_t indir_bytes = 0;

  rss_head.cmd = ETHTOOL_GRSSH;
  err = SendIoctl(ifname, (char *)&rss_head);

  if (err < 0 && errno == EOPNOTSUPP) {
    // do_srxfhindir
    struct ethtool_rxfh_indir indir_head;
    struct ethtool_rxfh_indir *indir;

    indir_head.cmd = ETHTOOL_GRXFHINDIR;
    indir_head.size = 0;

    err = SendIoctl(ifname, (char *)&indir_head);

    if (err < 0) {
      int error_number = errno;
      return absl::ErrnoToStatus(
          error_number,
          absl::StrFormat("Get indir_head error: %d", error_number));
    }
    indir = (struct ethtool_rxfh_indir *)malloc(
        sizeof(*indir) + indir_head.size * sizeof(*indir->ring_index));
    if (!indir) {
      return absl::ResourceExhaustedError(
          "Unable to allocate memory for rxfh_indir");
    }
    indir->cmd = ETHTOOL_SRXFHINDIR;
    indir->size = indir_head.size;
    // rxfhindir_equal
    for (int i = 0; i < indir->size; i++) {
      indir->ring_index[i] = (i % num_queues);
    }
    err = SendIoctl(ifname, (char *)indir);
    if (err < 0) {
      int error_number = errno;
      free(indir);
      return absl::ErrnoToStatus(
          error_number,
          absl::StrFormat("Cannot set RX flow hash indirection table: %d",
                          error_number));
    }
    free(indir);
    return absl::OkStatus();
  } else if (err < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        error_number,
        absl::StrFormat("Cannot get RX flow hash indir size and key size: %d",
                        error_number));
  }

  // do_srxfh
  indir_bytes = rss_head.indir_size * entry_size;
  rss = (struct ethtool_rxfh *)calloc(
      1, sizeof(*rss) + indir_bytes + rss_head.key_size);
  if (!rss) {
    return absl::ResourceExhaustedError("Unable to allocate memory for rss");
  }
  rss->cmd = ETHTOOL_SRSSH;
  rss->rss_context = 0;
  rss->hfunc = 0;
  rss->indir_size = rss_head.indir_size;
  rss->key_size = 0;
  for (int i = 0; i < rss->indir_size; i++) {
    rss->rss_config[i] = (i % num_queues);
  }
  err = SendIoctl(ifname, (char *)rss);
  if (err < 0) {
    int error_number = errno;
    free(rss);
    return absl::ErrnoToStatus(
        error_number,
        absl::StrFormat("Cannot set RX flow hash configuration: %d",
                        error_number));
  }
  free(rss);
  return absl::OkStatus();
}
absl::Status IoctlNicConfigurator::ControlEthtoolFeatures(
    const std::string &ifname, uint32_t ethtool_flag, bool enable) {
  struct feature_defs *defs;
  int any_changed = 0, any_mismatch = 0;
  uint32_t off_flags_wanted = 0;
  uint32_t off_flags_mask = 0;
  struct ethtool_sfeatures *efeatures = nullptr;
  struct feature_state *old_state = nullptr;
  struct feature_state *new_state = nullptr;
  struct ethtool_value eval;
  unsigned int i, j;
  int err;

  defs = GetFeatureDefs(ifname);
  if (!defs) {
    return absl::UnavailableError("Cannot get device feature names");
  }
  if (defs->n_features) {
    efeatures = (struct ethtool_sfeatures *)malloc(
        sizeof(*efeatures) + FEATURE_BITS_TO_BLOCKS(defs->n_features) *
                                 sizeof(efeatures->features[0]));
    if (!efeatures) {
      free(new_state);
      free(old_state);
      free(defs);
      free(efeatures);
      return absl::UnavailableError("Cannot parse arguments");
    }
    efeatures->cmd = ETHTOOL_SFEATURES;
    efeatures->size = FEATURE_BITS_TO_BLOCKS(defs->n_features);
    memset(efeatures->features, 0,
           FEATURE_BITS_TO_BLOCKS(defs->n_features) *
               sizeof(efeatures->features[0]));
  }

  // Enable the feature
  if (enable) {
    off_flags_wanted |= ethtool_flag;
  }

  off_flags_mask |= ethtool_flag;

  old_state = GetFeatures(ifname, defs);
  if (!old_state) {
    free(new_state);
    free(old_state);
    free(defs);
    free(efeatures);
    return absl::UnavailableError("Cannot get device feature state");
  }

  if (efeatures) {
    /* For each offload that the user specified, update any
     * related features that the user did not specify and that
     * are not fixed.  Warn if all related features are fixed.
     */
    for (i = 0; i < OFF_FLAG_DEF_SIZE; i++) {
      int fixed = 1;

      if (!(off_flags_mask & off_flag_def[i].value)) continue;

      for (j = 0; j < defs->n_features; j++) {
        if (defs->def[j].off_flag_index != (int)i ||
            !FEATURE_BIT_IS_SET(old_state->features.features, j, available) ||
            FEATURE_BIT_IS_SET(old_state->features.features, j, never_changed))
          continue;

        fixed = 0;
        if (!FEATURE_BIT_IS_SET(efeatures->features, j, valid)) {
          FEATURE_BIT_SET(efeatures->features, j, valid);
          if (off_flags_wanted & off_flag_def[i].value)
            FEATURE_BIT_SET(efeatures->features, j, requested);
        }
      }

      if (fixed)
        fprintf(stderr, "Cannot change %s\n", off_flag_def[i].long_name);
    }

    err = SendIoctl(ifname, (char *)efeatures);
    if (err < 0) {
      free(new_state);
      free(old_state);
      free(defs);
      free(efeatures);
      return absl::UnavailableError("Cannot set device feature settings");
    }
  } else {
    for (i = 0; i < OFF_FLAG_DEF_SIZE; i++) {
      if (!off_flag_def[i].set_cmd) continue;
      if (off_flags_mask & off_flag_def[i].value) {
        eval.cmd = off_flag_def[i].set_cmd;
        eval.data = !!(off_flags_wanted & off_flag_def[i].value);
        err = SendIoctl(ifname, (char *)&eval);
        if (err) {
          return absl::UnavailableError(absl::StrFormat(
              "Cannot set device %s settings", off_flag_def[i].long_name));
        }
      }
    }

    if (off_flags_mask & ETH_FLAG_EXT_MASK) {
      eval.cmd = ETHTOOL_SFLAGS;
      eval.data = (old_state->off_flags & ~off_flags_mask & ETH_FLAG_EXT_MASK);
      eval.data |= off_flags_wanted & ETH_FLAG_EXT_MASK;

      err = SendIoctl(ifname, (char *)&eval);
      if (err) {
        free(new_state);
        free(old_state);
        free(defs);
        free(efeatures);
        return absl::UnavailableError("Cannot set device flag settings");
      }
    }
  }

  /* Compare new state with requested state */
  new_state = GetFeatures(ifname, defs);
  if (!new_state) {
    free(new_state);
    free(old_state);
    free(defs);
    free(efeatures);
    return absl::UnavailableError(
        "Cannot get device feature state after requested changes");
  }
  any_changed = new_state->off_flags != old_state->off_flags;
  any_mismatch =
      (new_state->off_flags !=
       ((old_state->off_flags & ~off_flags_mask) | off_flags_wanted));
  for (i = 0; i < FEATURE_BITS_TO_BLOCKS(defs->n_features); i++) {
    if (new_state->features.features[i].active !=
        old_state->features.features[i].active)
      any_changed = 1;
    if (new_state->features.features[i].active !=
        ((old_state->features.features[i].active &
          ~efeatures->features[i].valid) |
         efeatures->features[i].requested))
      any_mismatch = 1;
  }
  if (any_mismatch) {
    if (!any_changed) {
      free(new_state);
      free(old_state);
      free(defs);
      free(efeatures);
      return absl::UnavailableError("Could not change any device features\n");
    }
  }

  free(new_state);
  free(old_state);
  free(defs);
  free(efeatures);

  return absl::OkStatus();
}

absl::Status IoctlNicConfigurator::SetNtuple(const std::string &ifname) {
  return ControlEthtoolFeatures(ifname, ETH_FLAG_NTUPLE, true);
}

absl::Status IoctlNicConfigurator::AddFlow(const std::string &ifname,
                                           const struct FlowSteerNtuple &ntuple,
                                           int queue_id, int location_id) {
  struct ethtool_rx_flow_spec rx_rule_fs;
  int err;

  memset(&rx_rule_fs, 0, sizeof(rx_rule_fs));
  rx_rule_fs.location = location_id;

  if (ntuple.flow_type == TCP_V4_FLOW) {
    memcpy(&rx_rule_fs.h_u.tcp_ip4_spec.ip4src, &ntuple.src_sin.sin_addr,
           sizeof(ntuple.src_sin.sin_addr));
    memcpy(&rx_rule_fs.h_u.tcp_ip4_spec.ip4dst, &ntuple.dst_sin.sin_addr,
           sizeof(ntuple.dst_sin.sin_addr));
    rx_rule_fs.m_u.tcp_ip4_spec.ip4src = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip4_spec.ip4dst = htonl(0xFFFFFFFF);
    memcpy(&rx_rule_fs.h_u.tcp_ip4_spec.psrc, &ntuple.src_sin.sin_port,
           sizeof(ntuple.src_sin.sin_port));
    memcpy(&rx_rule_fs.h_u.tcp_ip4_spec.pdst, &ntuple.dst_sin.sin_port,
           sizeof(ntuple.dst_sin.sin_port));
  } else if (ntuple.flow_type == TCP_V6_FLOW) {
    memcpy(&rx_rule_fs.h_u.tcp_ip6_spec.ip6src, &ntuple.src_sin6.sin6_addr,
           sizeof(ntuple.src_sin6.sin6_addr));
    memcpy(&rx_rule_fs.h_u.tcp_ip6_spec.ip6dst, &ntuple.dst_sin6.sin6_addr,
           sizeof(ntuple.dst_sin6.sin6_addr));
    rx_rule_fs.m_u.tcp_ip6_spec.ip6src[0] = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip6_spec.ip6src[1] = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip6_spec.ip6src[2] = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip6_spec.ip6src[3] = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip6_spec.ip6dst[0] = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip6_spec.ip6dst[1] = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip6_spec.ip6dst[2] = htonl(0xFFFFFFFF);
    rx_rule_fs.m_u.tcp_ip6_spec.ip6dst[3] = htonl(0xFFFFFFFF);
    memcpy(&rx_rule_fs.h_u.tcp_ip6_spec.psrc, &ntuple.src_sin6.sin6_port,
           sizeof(ntuple.src_sin.sin_port));
    memcpy(&rx_rule_fs.h_u.tcp_ip6_spec.pdst, &ntuple.dst_sin6.sin6_port,
           sizeof(ntuple.dst_sin.sin_port));
  }
  struct ethtool_rxnfc nfccmd;
  memset(&nfccmd, 0, sizeof(nfccmd));
  nfccmd.cmd = ETHTOOL_SRXCLSRLINS;
  nfccmd.fs = rx_rule_fs;
  nfccmd.fs.ring_cookie = queue_id;
  err = SendIoctl(ifname, (char *)&nfccmd);
  if (err < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        error_number, absl::StrFormat("Connnot insert RX class rule, error: %d",
                                      error_number));
  }
  return absl::OkStatus();
}

absl::Status IoctlNicConfigurator::RemoveFlow(const std::string &ifname,
                                              int location_id) {
  struct ethtool_rxnfc nfccmd;
  int err;

  nfccmd.cmd = ETHTOOL_SRXCLSRLDEL;
  nfccmd.fs.location = location_id;
  err = SendIoctl(ifname, (char *)&nfccmd);
  if (err < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        error_number, absl::StrFormat("Connnot delete RX class rule, error: %d",
                                      error_number));
  }
  return absl::OkStatus();
}
}  // namespace tcpdirect
