.PHONY: all clean

PROGRAMS = test_spmv_mmf

CPPFLAGS += -I../include -D_CHECK -DEPS=1e-8
LDFLAGS += -L../src
GPU_LDFLAGS := --linker-options "-lsparse -lgomp"

COMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS)
LINK = $(CXX) $(LDFLAGS)

%.o: %.cpp
	$(COMPILE) -c -o $@ $<

test_spmv_mmf: test_spmv_mmf.o
	$(LINK) $^ -o $@ -lsparse $(LIBS)

all: $(PROGRAMS)

clean:	
	$(RM) -f *.o *.s *~ *.optrpt $(PROGRAMS)
