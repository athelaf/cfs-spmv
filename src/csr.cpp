#include "cfs.hpp"
// Include internal header files
#include "matrix/csr_matrix.tpp"

namespace cfs {
namespace matrix {
namespace sparse {

// Explicit instantiations
template class CSRMatrix<int, float>;
template class CSRMatrix<int, double>;

} // end of namespace sparse
} // end of namespace matrix
} // end of namespace cfs
