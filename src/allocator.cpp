#include <cassert>
#include <iostream>

#include "utils/allocator.hpp"

namespace util {

void internal_free(void *pointer, Platform platform) {
  if (!pointer) {
    return;
  }

  if (platform == Platform::cpu) {
#ifdef _INTEL_COMPILER
    _mm_free(pointer);
#else
    free(pointer);
#endif
  }

  pointer = nullptr;
}

void *internal_alloc(size_t bytes, Platform platform) {
  void *pointer = NULL;

  if (platform == Platform::cpu) {
#ifdef _INTEL_COMPILER
    pointer = _mm_malloc(bytes, 64);
    if (!pointer) {
      std::cout << "[ERROR]: _mm_malloc() failed!" << std::endl;
      exit(1);
    }
#else
    if (posix_memalign(&pointer, 64, bytes) != 0) {
      std::cout << "[ERROR]: posix_memalign() failed!" << std::endl;
      exit(1);
    }
#endif
  }

  return pointer;
}

} // end of namespace util
