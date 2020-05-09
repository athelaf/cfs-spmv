#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([sparse], [0.8], [athena@cslab.ece.ntua.gr])
AC_CONFIG_SRCDIR([cfs_config.h.in])
AC_CONFIG_HEADERS([cfs_config.h])
AC_CONFIG_MACRO_DIR([m4check])
AC_CANONICAL_BUILD
AC_CANONICAL_HOST

dnl Locate the C++ compiler.
AC_LANG(C++)
AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory])

dnl Check for programs
AC_PROG_CXX([g++])
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_AR
AC_OPENMP

dnl Set additional compiler and linker flags among Makefiles
AC_SUBST(AM_CXXFLAGS, "$OPENMP_CXXFLAGS -Wall")
AC_SUBST(AM_CPPFLAGS, "$OPENMP_CXXFLAGS")

dnl Required dependencies
AC_CHECK_LIB([dl], [main])
AC_CHECK_LIB([interface], [main])
AC_CHECK_LIB([m], [main])
AC_CHECK_LIB([tbb], [main])

dnl Optional dependencies
dnl ----- Intel MKL
AC_ARG_WITH([mkl], AS_HELP_STRING([--with-mkl], [Benchmark Intel MKL. The value detect, which is the default, enables MKL if it can be found.]))
AS_IF([test "x$with_mkl" == "x"], [with_mkl="detect"])
AS_IF([test "x$with_mkl" == "xdetect"], [AX_CHECK_MKL],
      [test "x$with_mkl" == "xyes"], [AX_CHECK_MKL],
      [test "x$with_mkl" == "xno"], [],
      [AC_MSG_ERROR([invalid value $with_mkl for with_mkl.])]
)
AM_CONDITIONAL([BENCH_MKL], [test x$mkl_found = xtrue])

dnl --enable-dp option
AC_ARG_ENABLE([dp], AS_HELP_STRING([--enable-dp], [Compile SpMV benchmarks with double-precision floating-point values. Default is no]))
AS_IF([test "x$enable_dp" = "xyes"], [
  AC_DEFINE([_USE_DOUBLE], [1], [Define if logging is enabled.])
])

dnl --enable-log option
AC_ARG_ENABLE([log], AS_HELP_STRING([--enable-log], [Enable logging info. Default is no]))
AS_IF([test "x$enable_log" = "xyes"], [
  AC_DEFINE([_LOG_INFO], [1], [Define if logging is enabled.])
])

dnl Checks for header files
AC_CHECK_HEADERS([stdlib.h string.h])

dnl Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

dnl Checks for library functions.
AC_CHECK_FUNCS([memset strdup])

dnl Initialize automake
AM_INIT_AUTOMAKE([1.11 -Wall -Wno-extra-portability foreign])
LT_INIT([shared disable-static])

AC_CONFIG_FILES([Makefile
                 src/Makefile
		 test/Makefile
		 bench/Makefile])

AC_OUTPUT
