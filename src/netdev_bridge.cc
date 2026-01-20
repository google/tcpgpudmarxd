/*
 Copyright 2026 Google LLC

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

#include "include/netdev_bridge.h"

#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <linux/genetlink.h>
#include <time.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>

namespace gpudirect_tcpxd {

constexpr char kNetdevFamilyName[] = "netdev";
#define NETDEV_CMD_BIND_RX 13
#define NETDEV_CMD_BIND_TX 15

// Top-level attributes for NETDEV_CMD_BIND_RX as per the provided enum
#define NETDEV_A_DMABUF_IFINDEX 1
#define NETDEV_A_DMABUF_QUEUES 2
#define NETDEV_A_DMABUF_FD 3

// Attribute for the reply
#define NETDEV_A_DMABUF_ID 4

#define NETDEV_BIND_RX_REPLY_ATTR_MAX NETDEV_A_DMABUF_ID

// Attributes *within* the NETDEV_A_DMABUF_QUEUES nest
#define NETDEV_A_QUEUE_ID 1
#define NETDEV_A_QUEUE_TYPE 3

#define NETDEV_QUEUE_TYPE_RX 0

// Macro to iterate through the mnl nested attributes
// Resembles libmnl's mnl_attr_for_each_nested, but it
// fixes the void* cast issue for mnl_attr_get_payload.
#define mnl_for_each_nested_attr(attr, nest) \
  for ((attr) = (struct nlattr*) mnl_attr_get_payload(nest); \
       mnl_attr_ok((attr), (char *)mnl_attr_get_payload(nest) + \
       mnl_attr_get_payload_len(nest) - (char *)(attr)); \
       (attr) = mnl_attr_next(attr))

static int family_data_attr_cb(const struct nlattr* attr, void* data) {
  const struct nlattr** tb = (const struct nlattr**)data;
  int type = mnl_attr_get_type(attr);

  if (mnl_attr_type_valid(attr, CTRL_ATTR_MAX) < 0) return MNL_CB_OK;

  switch (type) {
    case CTRL_ATTR_FAMILY_NAME:
      if (mnl_attr_validate(attr, MNL_TYPE_STRING) < 0) {
        perror("mnl_attr_validate");
        return MNL_CB_ERROR;
      }
      break;
    case CTRL_ATTR_FAMILY_ID:
      if (mnl_attr_validate(attr, MNL_TYPE_U16) < 0) {
        perror("mnl_attr_validate");
        return MNL_CB_ERROR;
      }
      break;
    case CTRL_ATTR_VERSION:
    case CTRL_ATTR_HDRSIZE:
    case CTRL_ATTR_MAXATTR:
      if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) {
        perror("mnl_attr_validate");
        return MNL_CB_ERROR;
      }
      break;
    case CTRL_ATTR_OPS:
    case CTRL_ATTR_MCAST_GROUPS:
      if (mnl_attr_validate(attr, MNL_TYPE_NESTED) < 0) {
        perror("mnl_attr_validate");
        return MNL_CB_ERROR;
      }
      break;
  }
  tb[type] = attr;
  return MNL_CB_OK;
}

static int parse_family_ops_cb(const struct nlattr* attr, void* data) {
  const struct nlattr** tb = (const struct nlattr**)data;
  int type = mnl_attr_get_type(attr);

  if (mnl_attr_type_valid(attr, CTRL_ATTR_OP_MAX) < 0)
    return MNL_CB_OK;

  switch(type) {
  case CTRL_ATTR_OP_ID:
    if (mnl_attr_validate(attr, MNL_TYPE_U32) < 0) {
      perror("mnl_attr_validate");
      return MNL_CB_ERROR;
    }
    break;
  case CTRL_ATTR_OP_MAX:
    break;
  default:
    return MNL_CB_OK;
  }
  tb[type] = attr;
  return MNL_CB_OK;
}

static bool parse_genl_family_ops(struct nlattr* nest) {
  struct nlattr* attr;
  bool is_netdev_bind_rx = false;
  bool is_netdev_bind_tx = false;

  mnl_for_each_nested_attr(attr, nest) {
    struct nlattr* tb[CTRL_ATTR_OP_MAX+1] = {};
    mnl_attr_parse_nested(attr, parse_family_ops_cb, tb);
    uint32_t op_id = mnl_attr_get_u32(tb[CTRL_ATTR_OP_ID]);
    if (op_id == NETDEV_CMD_BIND_RX) {
      is_netdev_bind_rx = true;
    } else if (op_id == NETDEV_CMD_BIND_TX) {
      is_netdev_bind_tx = true;
    }
  }

  return is_netdev_bind_rx && is_netdev_bind_tx;
}

int family_cb(const struct nlmsghdr* nlh, void* data) {
  int* family_id = (int*)data;
  struct nlattr* tb[CTRL_ATTR_MAX + 1] = {};
  struct genlmsghdr* genlh = (struct genlmsghdr*)mnl_nlmsg_get_payload(nlh);
  mnl_attr_parse(nlh, sizeof(*genlh), family_data_attr_cb, tb);

  if (tb[CTRL_ATTR_FAMILY_ID]) {
    uint16_t netdev_family_id = mnl_attr_get_u16(tb[CTRL_ATTR_FAMILY_ID]);
    LOG(INFO) << "Init Family CB: Resolved Family ID for " << kNetdevFamilyName
              << " id " << netdev_family_id;
    *family_id = netdev_family_id;
  } else {
    LOG(ERROR) << "Init Family CB: Could not find Family ID attribute";
    return MNL_CB_ERROR;
  }

  if (tb[CTRL_ATTR_OPS]) {
    if (!parse_genl_family_ops(tb[CTRL_ATTR_OPS])) {
      LOG(ERROR) << "Init Family CB: Netdev ops not supported, "
                 << "removing family ID";
      // Reset familyID to ensure the netdev init fails
      // in case the netdev ops are not supported.
      *family_id = -1;
      return MNL_CB_ERROR;
    }
  }

  return MNL_CB_OK;
}

int bind_rx_reply_attr_cb(const struct nlattr* attr, void* data) {
  // tb is an array of nlattr pointers
  const struct nlattr** tb = (const struct nlattr**)data;
  int type = mnl_attr_get_type(attr);

  if (type > 0 && type <= NETDEV_BIND_RX_REPLY_ATTR_MAX) {
    tb[type] = attr;
  }
  return MNL_CB_OK;
}

int bind_rx_reply_cb(const struct nlmsghdr* nlh, void* data) {
  struct genlmsghdr* genlh = (struct genlmsghdr*)mnl_nlmsg_get_payload(nlh);
  struct nlattr* tb[NETDEV_BIND_RX_REPLY_ATTR_MAX + 1];
  int* p_netdev_family_id = (int*)data;

  memset(tb, 0, sizeof(tb));

  if (genlh->cmd != NETDEV_CMD_BIND_RX) {
    LOG(ERROR) << absl::StrFormat(
        "BindRx Reply CB: Ignoring unexpected message (type %u, cmd %u).",
        nlh->nlmsg_type, genlh->cmd);
    return MNL_CB_ERROR;
  }

  if (mnl_attr_parse(nlh, sizeof(*genlh), bind_rx_reply_attr_cb, tb) < 0) {
    LOG(ERROR) << "BindRx Reply CB: Failed to parse attributes in reply.";
    return MNL_CB_ERROR;
  }

  int* output_id = (int*)data;
  if (!tb[NETDEV_A_DMABUF_ID]) {
    LOG(ERROR) << "BindRx Reply CB: Cannot find the dmabuf id";
    return MNL_CB_ERROR;
  }
  LOG(INFO) << "BindRx Reply CB: got family id "
            << mnl_attr_get_u32(tb[NETDEV_A_DMABUF_ID]);
  *output_id = mnl_attr_get_u32(tb[NETDEV_A_DMABUF_ID]);
  return MNL_CB_STOP;
}

absl::Status NetdevBridge::Init() {
  struct mnl_socket* nl = mnl_socket_open(NETLINK_GENERIC);
  if (nl == NULL) {
    return absl::InternalError("Init: Cannot open netlink socket");
  }

  if (mnl_socket_bind(nl, 0, MNL_SOCKET_AUTOPID) < 0) {
    mnl_socket_close(nl);
    return absl::InternalError("Init: Cannot bind to netlink socket");
  }
  unsigned int portid = mnl_socket_get_portid(nl);

  char buf[MNL_SOCKET_BUFFER_SIZE];

  struct nlmsghdr* nlh = mnl_nlmsg_put_header(buf);
  nlh->nlmsg_type = GENL_ID_CTRL;
  nlh->nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
  nlh->nlmsg_seq = seq_;

  struct genlmsghdr* genlh =
      (struct genlmsghdr*)mnl_nlmsg_put_extra_header(nlh, sizeof(genlmsghdr));
  genlh->cmd = CTRL_CMD_GETFAMILY;
  genlh->version = 1;

  mnl_attr_put_strz(nlh, CTRL_ATTR_FAMILY_NAME, kNetdevFamilyName);

  int ret = mnl_socket_sendto(nl, nlh, nlh->nlmsg_len);
  if (ret < 0) {
    return absl::InternalError("Init: mnl socket send error");
  }

  ret = mnl_socket_recvfrom(nl, buf, sizeof(buf));
  if (ret < 0) {
    return absl::InternalError("Init: mnl socket recv error");
  }

  int family_id = -1;
  while (ret > 0) {
    ret = mnl_cb_run(buf, ret, seq_, portid, family_cb, &family_id);
    if (ret <= 0) break;
    ret = mnl_socket_recvfrom(nl, buf, sizeof(buf));
  }
  if (family_id == -1) {
    return absl::InternalError(
          "Init: failed to get netdev family with bind TX/RX ops");
  }

  nl_ = nl;
  netdev_family_id_ = family_id;

  return absl::OkStatus();
}

absl::StatusOr<int> NetdevBridge::BindRx(uint32_t ifindex,
                                         const std::vector<int>& queue_ids,
                                         uint32_t dmabuf_fd) {
  char buf[MNL_SOCKET_BUFFER_SIZE];

  struct nlmsghdr* nlh = mnl_nlmsg_put_header(buf);
  nlh->nlmsg_type = netdev_family_id_;
  nlh->nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
  nlh->nlmsg_seq = ++seq_;

  struct genlmsghdr* genlh =
      (struct genlmsghdr*)mnl_nlmsg_put_extra_header(nlh, sizeof(*genlh));
  genlh->cmd = NETDEV_CMD_BIND_RX;
  genlh->version = 1;

  mnl_attr_put_u32(nlh, NETDEV_A_DMABUF_IFINDEX, (uint32_t)ifindex);

  struct nlattr* queues_nest;

  for (size_t i = 0; i < queue_ids.size(); ++i) {
    if (queue_ids[i] < 0) {
      LOG(ERROR) << "BindRx: Invalid qid: " << queue_ids[i];
      continue;
    }

    queues_nest = mnl_attr_nest_start(nlh, NETDEV_A_DMABUF_QUEUES);
    if (!queues_nest) {
      return absl::InternalError(
          "BindRx: Failed to start attr nesting NETDEV_A_DMABUF_QUEUES");
    }
    mnl_attr_put_u32(nlh, NETDEV_A_QUEUE_ID, (uint32_t)queue_ids[i]);
    mnl_attr_put_u32(nlh, NETDEV_A_QUEUE_TYPE, NETDEV_QUEUE_TYPE_RX);

    mnl_attr_nest_end(nlh, queues_nest);
  }

  mnl_attr_put_u32(nlh, NETDEV_A_DMABUF_FD, dmabuf_fd);

  int ret = mnl_socket_sendto(nl_, nlh, nlh->nlmsg_len);
  if (ret < 0) {
    return absl::InternalError("BindRx: nl socket sendto failed");
  }

  unsigned int local_portid = mnl_socket_get_portid(nl_);
  int output_id = -1;

  ret = mnl_socket_recvfrom(nl_, buf, sizeof(buf));
  if (ret < 0) {
    return absl::InternalError("BindRx: nl socket recv failed");
  }

  while (ret > 0) {
    ret =
        mnl_cb_run(buf, ret, seq_, local_portid, bind_rx_reply_cb, &output_id);
    if (ret <= 0) break;
    ret = mnl_socket_recvfrom(nl_, buf, sizeof(buf));
  }

  if (output_id < 0) {
    return absl::InternalError("BindRx: Invalid recv output id");
  }

  return output_id;
}
void NetdevBridge::Cleanup() {
  LOG(INFO) << "NetdevBridge Cleanup";
  if (nl_ != nullptr) {
    mnl_socket_close(nl_);
    nl_ = nullptr;
  }
}
}  // namespace gpudirect_tcpxd