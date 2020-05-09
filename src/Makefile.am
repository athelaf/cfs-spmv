lib_LTLIBRARIES = libsparse.la
# Public library header files
include_HEADERS = $(srcdir)/../include/cfs.hpp $(srcdir)/../include/kernel/sparse_kernel.hpp $(srcdir)/../include/matrix/sparse_matrix.hpp $(srcdir)/../include/utils/platform.hpp 
# Internal source and header files
libsparse_la_SOURCES = allocator.hpp allocator.cpp cfs.cpp csr_matrix.hpp mmf.hpp mmf.cpp runtime.hpp runtime.cpp
libsparse_la_CPPFLAGS = -I$(srcdir)/../include -D_USE_BARRIER $(AM_CPPFLAGS)
