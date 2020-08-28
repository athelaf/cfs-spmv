#ifndef SPARSE_MATRIX_TPP
#define SPARSE_MATRIX_TPP

#include "csr_matrix.hpp"

namespace cfs {
namespace matrix {
namespace sparse {

template <typename IndexT, typename ValueT>
SparseMatrix<IndexT, ValueT>::~SparseMatrix() = default;

template <typename IndexT, typename ValueT>
SparseMatrix<IndexT, ValueT> *
SparseMatrix<IndexT, ValueT>::create(const string &filename, Format format,
                                     Platform platform) {
  if (format == Format::sss) {
    return new CSRMatrix<IndexT, ValueT>(filename, platform, true);
  } else if (format == Format::hyb) {
    return new CSRMatrix<IndexT, ValueT>(filename, platform, true, true);
  } else {
    return new CSRMatrix<IndexT, ValueT>(filename, platform);
  }
}

} // end of namespace sparse
} // end of namespace matrix
} // end of namespace cfs

#endif
