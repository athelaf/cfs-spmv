#pragma once

#include "platforms.hpp"

namespace util {

void internal_free(void *pointer, Platform platform = Platform::cpu);
void *internal_alloc(size_t bytes, Platform platform = Platform::cpu);

} // end of namespace util
