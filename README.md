# cfs-spmv
# Dependencies
* Boost Graph Library
* OpenMP
* Intel(R) Threading Building Blocks
* C++ compiler with c++14 support

# Compilation
Generic options include:
* LOG=1 to enable logging 
* USE_DOUBLE=1 to enable double-precision floating-point computations
* MKL=1 to enable benchmarking of Intel MKL (you may need to set MKL_ROOT to the correct path)
* RSB=1 to enable benchmarking of librsb (you may need to set RSB_ROOT to the correct path)

How to compile:
* GNU C/C++: make all
* Intel(R) C/C++: INTEL_ROOT=<path_to_intel_root> make -f Makefile.intel all
