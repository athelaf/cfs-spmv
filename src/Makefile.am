lib_LTLIBRARIES = libsparse.la
# Public library header files
include_HEADERS = $(srcdir)/../include/cfs.hpp $(srcdir)/../include/kernel/sparse_kernel.hpp $(srcdir)/../include/matrix/sparse_matrix.hpp $(srcdir)/../include/matrix/csr_matrix.hpp $(srcdir)/../include/utils/platform.hpp 
# Internal source and header files
libsparse_la_SOURCES = allocator.hpp csr_matrix.hpp mmf.hpp runtime.hpp sparse_kernel.tpp sparse_matrix.tpp allocator.cpp cfs.cpp csr.cpp mmf.cpp runtime.cpp
libsparse_la_CPPFLAGS = -I$(srcdir)/../include -D_USE_BARRIER $(AM_CPPFLAGS)
