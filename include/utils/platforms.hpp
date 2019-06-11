#pragma once

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

} // end of namespace util
