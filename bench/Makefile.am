bin_PROGRAMS = bench_spmv_mmf
bench_spmv_mmf_SOURCES = bench_spmv_mmf.cpp
bench_spmv_mmf_CPPFLAGS = -I$(srcdir)/../include
bench_spmv_mmf_LDADD = $(top_builddir)/src/.libs/libsparse.la $(AM_LDFLAGS)
if BENCH_MKL
bench_spmv_mmf_CPPFLAGS += -D_MKL
bench_spmv_mmf_LDADD += $(MKL_LIBS)
endif
