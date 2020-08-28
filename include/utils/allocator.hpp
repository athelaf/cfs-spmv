#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include "cfs_config.hpp"
#include "platform.hpp"

namespace cfs {
namespace util {
namespace memory {

void internal_free(void *pointer, Platform platform = Platform::cpu);
void *internal_alloc(size_t bytes, Platform platform = Platform::cpu);

} // end of namespace memory
} // end of namespace util
} // end of namespace cfs

#endif
