#ifndef _TCPDIRECT_COMMON_H_
#define _TCPDIRECT_COMMON_H_

namespace gpudirect_tcpxd {

constexpr int SO_DEVMEM_DONTNEED = 97;
constexpr int SO_DEVMEM_HEADER = 98;
constexpr int SO_DEVMEM_OFFSET = 99;
constexpr int SCM_DEVMEM_HEADER = SO_DEVMEM_HEADER;
constexpr int SCM_DEVMEM_OFFSET = SO_DEVMEM_OFFSET;

}  // namespace gpudirect_tcpxd

#endif
