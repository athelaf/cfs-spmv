#ifndef SPARSE_KERNEL_HPP
#define SPARSE_KERNEL_HPP

#include <cassert>
#include <iostream>

#include "cfs_config.hpp"
#include "matrix/sparse_matrix.hpp"

namespace cfs {

using namespace matrix::sparse;

namespace kernel {
namespace sparse {

template <typename IndexType, typename ValueType> struct SpDMV {
public:
  SpDMV() = delete;
  // Any preprocessing happens here
  SpDMV(SparseMatrix<IndexType, ValueType> *A, Tuning t = Tuning::Aggressive);
  void operator()(ValueType *__restrict y, const int M,
                  const ValueType *__restrict x, const int N);

private:
  SparseMatrix<IndexType, ValueType> *A_;
};

} // end of namespace sparse
} // end of namespace kernel
} // end of namespace cfs

#endif
