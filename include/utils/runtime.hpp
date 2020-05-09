#ifndef RUNTIME_HPP
#define RUNTIME_HPP

#include <atomic>
#include <iostream>
#include <sched.h>
#include <stdlib.h>

namespace util {
namespace runtime {

#ifndef _USE_BARRIER
#define MAX_THREADS 96

extern std::atomic<bool> done[MAX_THREADS][MAX_THREADS];
extern int counter[MAX_THREADS];
#endif

size_t get_threads();
void setaffinity_oncpu(unsigned int cpu);

} // end of namespace runtime
} // end of namespace util

#endif
