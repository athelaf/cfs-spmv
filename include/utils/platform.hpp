#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#include <cmath>

#include "cfs_config.hpp"

#ifdef _INTEL_COMPILER
#define PRAGMA_IVDEP _Pragma("ivdep")
#else
#define PRAGMA_IVDEP _Pragma("GCC ivdep")
//#define PRAGMA_IVDEP
#endif

namespace cfs {
namespace util {

using namespace std;

enum class Platform { cpu };
enum class Kernel { SpDMV };
enum class Tuning { None, Aggressive };
enum class Format { none, csr, sss, hyb };

inline int iceildiv(const int a, const int b) { return (a + b - 1) / b; }

inline bool isEqual(float x, float y) {
  const float epsilon = 1e-4;
  return abs(x - y) <= epsilon * abs(x);
  // see Knuth section 4.2.2 pages 217-218
}

inline bool isEqual(double x, double y) {
  const double epsilon = 1e-8;
  return abs(x - y) <= epsilon * abs(x);
  // see Knuth section 4.2.2 pages 217-218
}

} // end of namespace util
} // end of namespace cfs

#endif
