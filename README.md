# cfs-spmv
This is an implementation of the work presented in the paper:

"Athena Elafrou, Georgios Goumas, and Nectarios Koziris. 2019. Conflict-free symmetric sparse matrix-vector multiplication on multicore architectures. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC ’19)."

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
