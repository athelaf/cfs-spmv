# cfs-spmv
This is an implementation of the work presented in the paper:

"Athena Elafrou, Georgios Goumas, and Nectarios Koziris. 2019. Conflict-free symmetric sparse matrix-vector multiplication on multicore architectures. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC ’19)."

# Dependencies
* C++ compiler with c++11 support
* OpenMP
* Intel(R) Threading Building Blocks

# Compilation options
* --enable-log to enable logging 
* --enable-dp to enable double-precision floating-point computations

# Compilation
mkdir build; cd build; ../configure; make; make install