#include "cfs.hpp"
// Include internal header files
#include "kernel/sparse_kernel.tpp"
#include "matrix/sparse_matrix.tpp"

namespace cfs {
namespace matrix {
namespace sparse {

// Explicit instantiations
template class SparseMatrix<int, float>;
template class SparseMatrix<int, double>;

} // end of namespace sparse
} // end of namespace matrix

namespace kernel {
namespace sparse {

template struct SpDMV<int, float>;
template struct SpDMV<int, double>;

} // end of namespace sparse
} // end of namespace kernel
} // end of namespace cfs
