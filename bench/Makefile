.PHONY: all clean

PROGRAMS = bench_spmv_mmf_cpu

CPPFLAGS += -I../include
LDFLAGS += -L../src

ifeq ($(OPENBLAS), 1)
	CPPFLAGS += -I$(OPENBLAS_ROOT)/include -D_OPENBLAS
	LDFLAGS += -L$(OPENBLAS_ROOT)/lib
	LIBS += -lopenblas
endif

ifeq ($(RSB), 1)
	CPPFLAGS += $(shell librsb-config --I_opts) -D_RSB
	LIBS += $(shell librsb-config --static --ldflags)
endif

ifeq ($(MKL), 1)
	CPPFLAGS += -I${MKL_ROOT}/include -D_INTEL_MKL
	LDFLAGS += -L${MKL_ROOT}/lib/intel64
	LIBS += -Wl,--no-as-needed -lmkl_intel_lp64
ifeq ($(CXX), g++)
	LIBS += -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
else
	LIBS += -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
endif
endif

COMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS)
LINK = $(CXX) $(LDFLAGS)

%.o: %.cpp
	$(COMPILE) -c -o $@ $<

bench_spmv_mmf_cpu: bench_spmv_mmf_cpu.o
	$(LINK) $^ -o $@ -lsparse $(LIBS)

all: $(PROGRAMS)

clean:	
	$(RM) -f *.o *.s *~ *.optrpt $(PROGRAMS)
