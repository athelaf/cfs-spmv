#include "matrix/sparse_matrix.hpp"

namespace matrix {
namespace sparse {

#ifndef _USE_BARRIER
std::atomic<bool> done[MAX_THREADS][MAX_COLORS];
#endif

// Explicit instantiations
template class CSRMatrix<int, float>;
template class CSRMatrix<int, double>;

} // end of namespace sparse
} // end of namespace matrix
