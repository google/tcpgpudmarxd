# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(PARENT_DIR ${SELF_DIR} DIRECTORY)

include(${SELF_DIR}/gpudirect_tcpxd.cmake)

set(gpudirect_tcpxd_LIB_DIR ${SELF_DIR}/lib)
set(gpudirect_tcpxd_INCLUDE_DIR ${SELF_DIR} ${PARENT_DIR})
set(gpudirect_tcpxd_LIBRARIES rx_rule_client unix_socket_client_lib proto_utils socket_helper)
