#ifndef SPARSE_KERNEL_HPP
#define SPARSE_KERNEL_HPP

#include "matrix/sparse_matrix.hpp"

namespace kernel {
namespace sparse {

using namespace matrix::sparse;

template <typename IndexType, typename ValueType> struct SpDMV {
public:
  SpDMV() = delete;

  // Any preprocessing happens here
  SpDMV(SparseMatrix<IndexType, ValueType> *A, Tuning t = Tuning::Aggressive)
      : A_(A) {
    bool ret = A_->tune(Kernel::SpDMV, t);
    if (ret) {
#ifdef _LOG_INFO
      std::cout << "[INFO]: matrix format was tuned successfully" << std::endl;
#endif
    }
  }

  void operator()(ValueType *__restrict y, const int M,
                  const ValueType *__restrict x, const int N) {
    assert(A_->nrows() == M);
    assert(A_->ncols() == N);
    A_->dense_vector_multiply(y, x);
  }

private:
  SparseMatrix<IndexType, ValueType> *A_;
};

} // end of namespace sparse
} // end of namespace kernel

#endif
