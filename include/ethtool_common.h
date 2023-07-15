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

#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_COMMON_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_COMMON_H_

#include <linux/ethtool.h>
#include <stdint.h>

#include <cstddef>
namespace tcpdirect {
#define KERNEL_VERSION(a, b, c) (((a) << 16) + ((b) << 8) + (c))

inline static constexpr int OFF_FLAG_DEF_SIZE = 42;
struct feature_def {
  char name[ETH_GSTRING_LEN];
  int off_flag_index; /* index in off_flag_def; negative if none match */
};
struct feature_defs {
  size_t n_features;
  /* Number of features each offload flag is associated with */
  unsigned int off_flag_matched[OFF_FLAG_DEF_SIZE];
  /* Name and offload flag index for each feature */
  struct feature_def def[0];
};

struct feature_state {
  uint32_t off_flags;
  struct ethtool_gfeatures features;
};

#define DIV_ROUND_UP(n, d) (((n) + (d)-1) / (d))

#define FEATURE_BITS_TO_BLOCKS(n_bits) DIV_ROUND_UP(n_bits, 32U)
#define FEATURE_WORD(blocks, index, field) ((blocks)[(index) / 32U].field)
#define FEATURE_FIELD_FLAG(index) (1U << (index) % 32U)
#define FEATURE_BIT_SET(blocks, index, field) \
  (FEATURE_WORD(blocks, index, field) |= FEATURE_FIELD_FLAG(index))
#define FEATURE_BIT_CLEAR(blocks, index, field) \
  (FEATURE_WORD(blocks, index, filed) &= ~FEATURE_FIELD_FLAG(index))
#define FEATURE_BIT_IS_SET(blocks, index, field) \
  (FEATURE_WORD(blocks, index, field) & FEATURE_FIELD_FLAG(index))

struct off_flag_def {
  const char* short_name;
  const char* long_name;
  const char* kernel_name;
  uint32_t get_cmd, set_cmd;
  uint32_t value;
  /* For features exposed through ETHTOOL_GFLAGS, the oldest
   * kernel version for which we can trust the result.  Where
   * the flag was added at the same time the kernel started
   * supporting the feature, this is 0 (to allow for backports).
   * Where the feature was supported before the flag was added,
   * it is the version that introduced the flag.
   */
  uint32_t min_kernel_ver;
};

/* Internal values for old-style offload flags.  Values and names
 * must not clash with the flags defined for ETHTOOL_{G,S}FLAGS.
 */
#define ETH_FLAG_RXCSUM (1 << 0)
#define ETH_FLAG_TXCSUM (1 << 1)
#define ETH_FLAG_SG (1 << 2)
#define ETH_FLAG_TSO (1 << 3)
#define ETH_FLAG_UFO (1 << 4)
#define ETH_FLAG_GSO (1 << 5)
#define ETH_FLAG_GRO (1 << 6)
#define ETH_FLAG_INT_MASK                                           \
  (ETH_FLAG_RXCSUM | ETH_FLAG_TXCSUM | ETH_FLAG_SG | ETH_FLAG_TSO | \
   ETH_FLAG_UFO | ETH_FLAG_GSO | ETH_FLAG_GRO),
/* Mask of all flags defined for ETHTOOL_{G,S}FLAGS. */
#define ETH_FLAG_EXT_MASK                                               \
  (ETH_FLAG_LRO | ETH_FLAG_RXVLAN | ETH_FLAG_TXVLAN | ETH_FLAG_NTUPLE | \
   ETH_FLAG_RXHASH)

const struct off_flag_def off_flag_def[] = {
    {"rx", "rx-checksumming", "rx-checksum", ETHTOOL_GRXCSUM, ETHTOOL_SRXCSUM,
     ETH_FLAG_RXCSUM, 0},
    {"tx", "tx-checksumming", "tx-checksum-*", ETHTOOL_GTXCSUM, ETHTOOL_STXCSUM,
     ETH_FLAG_TXCSUM, 0},
    {"sg", "scatter-gather", "tx-scatter-gather*", ETHTOOL_GSG, ETHTOOL_SSG,
     ETH_FLAG_SG, 0},
    {"tso", "tcp-segmentation-offload", "tx-tcp*-segmentation", ETHTOOL_GTSO,
     ETHTOOL_STSO, ETH_FLAG_TSO, 0},
    {"ufo", "udp-fragmentation-offload", "tx-udp-fragmentation", ETHTOOL_GUFO,
     ETHTOOL_SUFO, ETH_FLAG_UFO, 0},
    {"gso", "generic-segmentation-offload", "tx-generic-segmentation",
     ETHTOOL_GGSO, ETHTOOL_SGSO, ETH_FLAG_GSO, 0},
    {"gro", "generic-receive-offload", "rx-gro", ETHTOOL_GGRO, ETHTOOL_SGRO,
     ETH_FLAG_GRO, 0},
    {"lro", "large-receive-offload", "rx-lro", 0, 0, ETH_FLAG_LRO,
     KERNEL_VERSION(2, 6, 24)},
    {"rxvlan", "rx-vlan-offload", "rx-vlan-hw-parse", 0, 0, ETH_FLAG_RXVLAN,
     KERNEL_VERSION(2, 6, 37)},
    {"txvlan", "tx-vlan-offload", "tx-vlan-hw-insert", 0, 0, ETH_FLAG_TXVLAN,
     KERNEL_VERSION(2, 6, 37)},
    {"ntuple", "ntuple-filters", "rx-ntuple-filter", 0, 0, ETH_FLAG_NTUPLE, 0},
    {"rxhash", "receive-hashing", "rx-hashing", 0, 0, ETH_FLAG_RXHASH, 0},
};

}  // namespace tcpdirect
#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_COMMON_H_
