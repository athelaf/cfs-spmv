.PHONY: all clean

CPPFLAGS += -I../include
COMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS)
LINK = $(CXX) $(LDFLAGS)

SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:%.cpp=%.o)

coo_matrix.o: coo_matrix.cpp
	$(COMPILE) -c -o $@ $<
csr_matrix.o: csr_matrix.cpp
	$(COMPILE) -c -o $@ $<

%.o: %.cpp
	$(COMPILE) -c -o $@ $<

libsparse.so: $(OBJECTS)
	$(LINK) -shared $^ -o $@

all: $(OBJECTS) libsparse.so

clean:
	$(RM) -f *.o *.s *.so *~ *.optrpt
