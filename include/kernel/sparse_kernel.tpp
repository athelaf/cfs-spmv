#ifndef SPARSE_KERNEL_TPP
#define SPARSE_KERNEL_TPP

namespace cfs {
namespace kernel {
namespace sparse {

template <typename IndexType, typename ValueType>
SpDMV<IndexType, ValueType>::SpDMV(SparseMatrix<IndexType, ValueType> *A,
                                   Tuning t)
    : A_(A) {
  bool ret = A_->tune(Kernel::SpDMV, t);
  if (ret) {
#ifdef _LOG_INFO
    std::cout << "[INFO]: matrix format was tuned successfully" << std::endl;
#endif
  }
}

template <typename IndexType, typename ValueType>
void SpDMV<IndexType, ValueType>::
operator()(ValueType *__restrict y, const int M, const ValueType *__restrict x,
           const int N) {
  assert(A_->nrows() == M);
  assert(A_->ncols() == N);
  A_->dense_vector_multiply(y, x);
}

} // end of namespace sparse
} // end of namespace kernel
} // end of namespace cfs

#endif
