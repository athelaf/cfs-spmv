#include "matrix/sparse_matrix.hpp"

namespace matrix {
namespace sparse {

// Explicit instantiations
template class COOMatrix<int, float>;
template class COOMatrix<int, double>;

} // end of namespace sparse
} // end of namespace matrix
