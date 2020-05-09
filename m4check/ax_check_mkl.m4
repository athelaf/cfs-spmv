dnl
dnl ax_check_mkl.m4 -- Macros related to the Intel MKL library installation
dnl
dnl Copyright (C) 2020, Computing Systems Laboratory (CSLab), NTUA
dnl Copyright (C) 2020, Athena Elafrou
dnl All rights reserved.
dnl
dnl This file is distributed under the BSD License. See LICENSE.txt for details.
dnl

AC_DEFUN([AX_CHECK_MKL],
[
	AC_CHECK_HEADERS([mkl.h])	
	AC_MSG_CHECKING([for -lmkl_core])
	OLD_LIBS="$LIBS"
	LIBS="$LIBS -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
	AC_LINK_IFELSE(
		[AC_LANG_PROGRAM([
			#include <stdio.h>
			#include <mkl.h>
			],
			[
				printf("%d.%d.%d\n", __INTEL_MKL__, __INTEL_MKL_MINOR__, __INTEL_MKL_UPDATE__);
			])],
		[AC_MSG_RESULT([yes]); mkl_found=true; break;],
		[AC_MSG_RESULT([no]); mkl_found=false; break;]
	)
	LIBS="$OLD_LIBS"

	if test x$mkl_found = xtrue; then
           AC_DEFINE([_MKL], [1], [Build SpMV benchmarks with MKL.])
	   MKL_LIBS="-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core"
	fi

	AC_SUBST([MKL_LIBS])
])
