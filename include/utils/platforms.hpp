#pragma once

#include <cmath>

#ifdef _INTEL_COMPILER
#define PRAGMA_IVDEP _Pragma("ivdep")
#else
#define PRAGMA_IVDEP _Pragma("GCC ivdep")
//#define PRAGMA_IVDEP
#endif

namespace util {

enum class Platform { cpu };
enum class Kernel { SpDMV };
enum class Tuning { None, Aggressive };
enum class Format { coo, csr, sss, hyb };

inline int iceildiv(const int a, const int b) { return (a + b - 1) / b; }
inline bool isEqual(double x, double y) {
  const double epsilon = 1e-7 /* some small number such as 1e-5 */;
  return abs(x - y) <= epsilon * abs(x);
  // see Knuth section 4.2.2 pages 217-218
}

} // end of namespace util
