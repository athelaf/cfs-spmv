#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_set>

// Third-party libraries
#include <omp.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_vector.h>
#ifdef _METIS
#include <metis.h>
#endif
#ifdef _KAHIP
#include <kaHIP_interface.h>
#endif

#include "cfs_config.hpp"
#include "io/mmf.hpp"
#include "utils/allocator.hpp"
#include "utils/platform.hpp"
#include "utils/runtime.hpp"

using namespace std;

namespace cfs {

using namespace io;
using namespace util::memory;
using namespace util::runtime;

namespace matrix {
namespace sparse {

// Forward declarations
template <typename IndexT, typename ValueT> class SparseMatrix;

template <typename IndexT, typename ValueT>
class CSRMatrix : public SparseMatrix<IndexT, ValueT> {
public:
  CSRMatrix() = delete;
  // Initialize CSR from an MMF file
  CSRMatrix(const string &filename, Platform platform = Platform::cpu,
            bool symmetric = false, bool hybrid = false);
  // Initialize CSR from another CSR matrix (no ownership)
  CSRMatrix(IndexT *rowptr, IndexT *colind, ValueT *values, IndexT nrows,
            IndexT ncols, bool symmetric = false, bool hybrid = false,
            Platform platform = Platform::cpu);
  virtual ~CSRMatrix();
  virtual int nrows() const override { return nrows_; }
  virtual int ncols() const override { return ncols_; }
  virtual int nnz() const override { return nnz_; }
  virtual bool symmetric() const override { return symmetric_; }
  virtual size_t size() const override;
  virtual inline Platform platform() const override { return platform_; }
  virtual bool tune(Kernel k, Tuning t) override;
  virtual void dense_vector_multiply(ValueT *__restrict y,
                                     const ValueT *__restrict x) override {
    spmv_fn(y, x);
  }

  // Export CSR internal representation
  IndexT *rowptr() const { return rowptr_; }
  IndexT *colind() const { return colind_; }
  ValueT *values() const { return values_; }

private:
  // Forward and alias declarations of helper classes
  class SymThreadData;
  struct ConflictMap;
  struct WeightedVertex;
  struct DecreasingVertexWeight;
  using ConflictGraph =
      tbb::concurrent_vector<tbb::concurrent_unordered_set<IndexT>>;

  /*
   * Tunable parameters
   */
  static constexpr int BlkBits = 4;
  static constexpr int BlkFactor = 1 << BlkBits;
  static constexpr int BalancingSteps = 10;
  static constexpr int HybBwThreshold = 10000;
  static constexpr double ImbalanceTol = 0.0;
  static constexpr int MaxColors = MaxThreads;

  Platform platform_;
  int nrows_, ncols_, nnz_;
  bool symmetric_, owns_data_;
  IndexT *rowptr_;
  IndexT *colind_;
  ValueT *values_;
  // Hybrid
  bool hybrid_;
  bool split_by_bw_;
  int nnz_lbw_, nnz_hbw_;
  IndexT *rowptr_h_;
  IndexT *colind_h_;
  ValueT *values_h_;
  // Partitioning
  bool part_by_nrows_, part_by_nnz_, part_by_ncnfls_;
  int nthreads_;
  int *row_split_, *row_part_;
  vector<vector<int>> row_subset_;
  // Symmetry compression
  bool cmp_symmetry_;
  bool atomics_, effective_ranges_, local_vectors_indexing_;
  bool conflict_free_apriori_, conflict_free_aposteriori_;
  int nnz_low_, nnz_diag_;
  int ncnfls_, ncolors_, nranges_;
  ConflictMap *cnfl_map_;
  ValueT **y_local_;
  SymThreadData **sym_thread_data_;
  // Internal function pointers
  function<void(ValueT *__restrict, const ValueT *__restrict)> spmv_fn;

  /*
  * Preprocessing routines
  */
  // Decomposes matrix into low-bandwidth and high-bandwidth submatrices
  void split_by_bandwidth();
  // Assigns chunks of consecutive rows to threads so that each thread has
  // approximately the same number of rows
  void partition_by_nrows(int nthreads);
  // Assigns chunks of consecutive rows to threads so that each thread has
  // approximately the same number of nonzeros
  void partition_by_nnz(int nthreads);
#if defined(_METIS) || defined(_KAHIP)
  // Assigns rows to threads so as to reduce the number of direct conflicts for
  // symmetric SpMV
  void partition_by_conflicts(int nthreads);
#endif

  /*
   * Symmetry compression
   */
  void compress_symmetry();
  void serial();
  // Method 1: atomics
  void atomics();
  // Method 2: local vectors with effective ranges
  void effective_ranges();
  // Method 3: local vectors indexing
  void local_vectors_indexing();
  // Method 4: conflict-free a priori
  void conflict_free_apriori();
  // Method 5: conflict-free a posteriori
  void conflict_free_aposteriori();
  void estimate_imbalance();
  // Common utilities for methods 4 & 5
  // Graph coloring + load balancing
  void color_greedy(const ConflictGraph &g, const vector<WeightedVertex> &v,
                    bool balance, vector<int> &color);
  void color_dsatur(const ConflictGraph &g, const vector<WeightedVertex> &v,
                    bool balance, vector<int> &color);
  // Parallel graph coloring
  void parallel_color(const ConflictGraph &g,
                      tbb::concurrent_vector<int> &color);
  // Parallel graph coloring + load balancing
  void parallel_color(const ConflictGraph &g, const vector<WeightedVertex> &v,
                      tbb::concurrent_vector<int> &color);
  void ordering_heuristic(const ConflictGraph &g, vector<int> &order);
  void largest_first(const ConflictGraph &g, vector<int> &order);
  void first_fit_round_robin(const ConflictGraph &g, vector<int> &order);

  /*
   * Sparse Matrix - Dense Vector Multiplication kernels
   */
  void cpu_mv_serial(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv(ValueT *__restrict y, const ValueT *__restrict x);

  // Symmetric kernels
  void cpu_mv_sym_serial(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_sym_atomics(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_sym_effective_ranges(ValueT *__restrict y,
                                   const ValueT *__restrict x);
  void cpu_mv_sym_local_vectors_indexing(ValueT *__restrict y,
                                         const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free_apriori(ValueT *__restrict y,
                                        const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free_v1(ValueT *__restrict y,
                                   const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free_v2(ValueT *__restrict y,
                                   const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free_hyb_v1(ValueT *__restrict y,
                                       const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free_hyb_v2(ValueT *__restrict y,
                                       const ValueT *__restrict x);

  // Helper nested structs/classes for symmetry compression
  struct ConflictMap {
    int length;
    short *cpu;
    int *pos;
  };

  struct WeightedVertex {
    int vid;
    int tid;
    int weight;
    WeightedVertex() : vid(0), tid(0), weight(0) {}
    WeightedVertex(int vertex_id, int thread_id, int weight)
        : vid(vertex_id), tid(thread_id), weight(weight) {}
  };

  struct DecreasingVertexWeight {
    bool operator()(const WeightedVertex &lhs, const WeightedVertex &rhs) {
      return lhs.weight > rhs.weight;
    }
  };

  class SymThreadData {
  public:
    int nrows_;         // Number of rows in lower triangular part
    IndexT row_offset_; // Row offset of this partition
    IndexT *rowptr_;    // Row pointer array for lower triangular part
    IndexT *rowind_;    // Mapping of local to global row index
    IndexT *colind_;    // Column index array for lower triangular part
    ValueT *values_;    // Values array for lower triangular part
    ValueT *diagonal_;  // Values of diagonal elements (padded)
    IndexT *range_ptr_; // Number of ranges per color
    IndexT *range_start_, *range_end_; // Ranges start/end
    IndexT *rowptr_h_; // Row pointer array for high-bandwidth part
    IndexT *colind_h_; // Column index array for high-bandwidth part
    ValueT *values_h_; // Values array for high-bandwidth part
    vector<int> deps_[MaxColors];
    // Other information
    int nranges_;  // Total number of ranges
    int nnz_;      // Total number of nonzeros
    int nnz_low_;  // Number of nonzeros in lower triangular part
    int nnz_diag_; // Number of nonzeros in diagonal
    int nnz_hbw_;  // Number of nonzeros in high-bandwidth part
    // Fields for other methods
    IndexT *color_ptr_;
    ValueT *local_vector_;    // Local vector for reduction-based methods
    int map_start_, map_end_; // Conflict map start/end
    // Platform-specific data
    Platform platform_;

    SymThreadData()
        : rowptr_(nullptr), rowind_(nullptr), colind_(nullptr),
          values_(nullptr), diagonal_(nullptr), range_ptr_(nullptr),
          range_start_(nullptr), range_end_(nullptr), rowptr_h_(nullptr),
          colind_h_(nullptr), values_h_(nullptr), color_ptr_(nullptr),
          local_vector_(nullptr), platform_(Platform::cpu) {}

    SymThreadData(Platform platform)
        : rowptr_(nullptr), rowind_(nullptr), colind_(nullptr),
          values_(nullptr), diagonal_(nullptr), range_ptr_(nullptr),
          range_start_(nullptr), range_end_(nullptr), rowptr_h_(nullptr),
          colind_h_(nullptr), values_h_(nullptr), color_ptr_(nullptr),
          local_vector_(nullptr), platform_(platform) {}

    ~SymThreadData() {
      internal_free(rowptr_, platform_);
      internal_free(colind_, platform_);
      internal_free(values_, platform_);
      internal_free(rowptr_h_, platform_);
      internal_free(colind_h_, platform_);
      internal_free(values_h_, platform_);
      internal_free(diagonal_, platform_);
      internal_free(color_ptr_, platform_);
      internal_free(local_vector_, platform_);
      internal_free(range_ptr_, platform_);
      internal_free(range_start_, platform_);
      internal_free(range_end_, platform_);
    }
  };
};

} // end of namespace sparse
} // end of namespace matrix
} // end of namespace cfs

#endif
