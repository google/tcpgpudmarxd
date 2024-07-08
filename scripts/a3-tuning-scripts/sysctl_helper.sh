#!/bin/bash

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set_and_verify() {
  local -r file="$1"
  local -r expected="$2"

  echo "${expected}" > "${file}"
  local -r value="$(< "${file}")"

  if [[ "$?" -ne 0 ]]; then
    logger "type=error file=\"${file}\""
    ((GLOBAL_NETBASE_ERROR_COUNTER+=1))
  else
    if [[ "${value}" != "${expected}" ]]; then
      logger "type=diff file=\"${file}\" value=${value} expected=${expected}"
      ((GLOBAL_NETBASE_ERROR_COUNTER+=1))
    fi
  fi
}

# Sets the file to some expected value, unless its
# current value is already larger than the expected value.
set_if_lt() {
  local -r file="$1"
  local -r expected="$2"

  local -r actual=$(cat "${file}")

  if [[ "${expected}" -gt "${actual}" ]]; then
    set_and_verify "${file}" "${expected}"
  else
    logger "skip setting file=\"${file}\" to smaller value=\"${expected}\", current value=\"${actual}\""
  fi
}
