#pragma once

#include <iostream>
#include <sched.h>
#include <stdlib.h>

namespace util {
namespace runtime {

size_t get_threads();
void setaffinity_oncpu(unsigned int cpu);

} // end of namespace runtime
} // end of namespace util
