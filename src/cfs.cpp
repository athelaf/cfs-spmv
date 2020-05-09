#include "matrix/sparse_matrix.hpp"

namespace matrix {
namespace sparse {

// Explicit instantiations
template class CSRMatrix<int, float>;
template class CSRMatrix<int, double>;
template class SparseMatrix<int, float>;
template class SparseMatrix<int, double>;

} // end of namespace sparse
} // end of namespace matrix
