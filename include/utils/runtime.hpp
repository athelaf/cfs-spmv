#ifndef RUNTIME_HPP
#define RUNTIME_HPP

#include <atomic>
#include <iostream>
#include <sched.h>
#include <stdlib.h>

#include "cfs_config.hpp"

namespace cfs {
namespace util {
namespace runtime {

const int MaxThreads = 96;

#ifndef _USE_BARRIER
extern std::atomic<bool> done[MaxThreads][MaxThreads];
extern int counter[MaxThreads];
#endif

size_t get_num_threads();
void setaffinity_oncpu(unsigned int cpu);

} // end of namespace runtime
} // end of namespace util
} // end of namespace cfs

#endif
