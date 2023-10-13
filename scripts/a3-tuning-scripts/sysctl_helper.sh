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