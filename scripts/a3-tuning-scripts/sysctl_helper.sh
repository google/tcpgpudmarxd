#!/bin/bash
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
