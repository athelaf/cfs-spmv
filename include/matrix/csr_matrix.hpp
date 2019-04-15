#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/bandwidth.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/graph/smallest_last_ordering.hpp>
#include <boost/property_map/shared_array_property_map.hpp>
#include <boost/ref.hpp>

#include <omp.h>

#include "io/mmf.hpp"
#include "utils/allocator.hpp"
#include "utils/platforms.hpp"
#include "utils/runtime.hpp"

using namespace std;
using namespace util;
using namespace util::io;
using namespace util::runtime;

namespace matrix {
namespace sparse {

// Formward declarations
template <typename IndexT, typename ValueT> class SparseMatrix;

struct ConflictMap {
  int length;
  short *cpu;
  int *pos;
};

// FIXME convert to nested classes
template <typename IndexT, typename ValueT> struct SymmetryCompressionData {
  int nrows_;            // Number of rows
  int nnz_lower_;        // Number of nonzeros in lower triangular part
  int nnz_diag_;         // Number of nonzeros in diagonal
  IndexT *rowptr_;       // Row pointer array for lower triangular part
  IndexT *colind_;       // Column index array for lower triangular part
  ValueT *values_;       // Values array for lower triangular part
  ValueT *diagonal_;     // Values of diagonal elements (padded)
  ValueT *local_vector_; // Local vector for reduction-based methods
  IndexT *range_ptr_;    // Number of ranges per color
  IndexT *range_start_, *range_end_; // Ranges start/end
  int map_start_, map_end_;          // Conflict map start/end
  int nranges_;                      // Total number or ranges
  int ncolors_;                      // Number or colors
  IndexT *rowind_;
  int nrows_left_;
  Platform platform_;

public:
  SymmetryCompressionData()
      : rowptr_(nullptr), colind_(nullptr), values_(nullptr),
        diagonal_(nullptr), local_vector_(nullptr), range_ptr_(nullptr),
        range_start_(nullptr), range_end_(nullptr), ncolors_(0),
        platform_(Platform::cpu) {}

  SymmetryCompressionData(Platform platform)
      : rowptr_(nullptr), colind_(nullptr), values_(nullptr),
        diagonal_(nullptr), local_vector_(nullptr), range_ptr_(nullptr),
        range_start_(nullptr), range_end_(nullptr), ncolors_(0),
        platform_(platform) {}

  ~SymmetryCompressionData() {
    internal_free(rowptr_, platform_);
    internal_free(colind_, platform_);
    internal_free(values_, platform_);
    internal_free(diagonal_, platform_);
    internal_free(local_vector_, platform_);
    internal_free(range_ptr_, platform_);
    internal_free(range_start_, platform_);
    internal_free(range_end_, platform_);
  }
};

struct WeightedVertex {
  int nnz;
  int tid;
};

template <typename IndexT, typename ValueT>
class CSRMatrix : public SparseMatrix<IndexT, ValueT> {
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                                WeightedVertex>
      ColoringGraph;
  typedef boost::graph_traits<ColoringGraph>::vertex_descriptor Vertex;
  typedef boost::graph_traits<ColoringGraph>::vertices_size_type
      vertices_size_type;
  typedef boost::property_map<ColoringGraph, boost::vertex_index_t>::const_type
      vertex_index_map;
  typedef boost::iterator_property_map<vertices_size_type *, vertex_index_map>
      ColorMap;

public:

  CSRMatrix() = delete;

  // Initialize CSR from an MMF file
  CSRMatrix(const string &filename, Platform platform = Platform::cpu, bool symmetric = false)
      : platform_(platform), owns_data_(true) {
    MMF<IndexT, ValueT> mmf(filename);
    symmetric_ = mmf.IsSymmetric();
    if (!symmetric) {
      symmetric_ = false;
#ifdef _LOG_INFO
      cout << "[INFO]: using CSR format to store the sparse matrix..." << endl;
#endif
    }
    if (symmetric) {
      if (symmetric_ != symmetric) {
#ifdef _LOG_INFO
	cout << "[INFO]: matrix is not symmetric!" << endl;
	cout << "[INFO]: rolling back to CSR format..." << endl;      
#endif
      } else {
#ifdef _LOG_INFO
	cout << "[INFO]: using SSS format to store the sparse matrix..." << endl;
#endif
      }
    }
    nrows_ = mmf.GetNrRows();
    ncols_ = mmf.GetNrCols();
    nnz_ = mmf.GetNrNonzeros();
    rowptr_ =
        (IndexT *)internal_alloc((nrows_ + 1) * sizeof(IndexT), platform_);
    colind_ = (IndexT *)internal_alloc(nnz_ * sizeof(IndexT), platform_);
    values_ = (ValueT *)internal_alloc(nnz_ * sizeof(ValueT), platform_);
    // Partitioning
    split_nnz_ = false;
    nthreads_ = get_threads();
    row_split_ = nullptr;
    // Symmetry compression
    cmp_symmetry_ = atomics_ = effective_ranges_ = explicit_conflicts_ = conflict_free_ = false;
    nnz_lower_ = nnz_diag_ = nrows_left_ = nconflicts_ = ncolors_ = 0;
    rowptr_sym_ = colind_sym_ = nullptr;
    values_sym_ = diagonal_ = nullptr;

    // Enforce first touch policy
    #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
    for (int i = 0; i < nrows_ + 1; i++) {
      rowptr_[i] = 0;
    }
    #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
    for (int i = 0; i < nnz_; i++) {
      colind_[i] = 0;
      values_[i] = 0.0;
    }

    auto iter = mmf.begin();
    auto iter_end = mmf.end();
    IndexT row_i = 0, val_i = 0, row_prev = 0;
    IndexT row, col;
    ValueT val;

    rowptr_[row_i++] = val_i;
    for (; iter != iter_end; ++iter) {
      // MMF returns one-based indices
      row = (*iter).row - 1;
      col = (*iter).col - 1;
      val = (*iter).val;
      assert(row >= row_prev);
      assert(row < nrows_);
      assert(col >= 0 && col < ncols_);
      assert(val_i < nnz_);

      if (row != row_prev) {
        for (IndexT i = 0; i < row - row_prev; i++) {
          rowptr_[row_i++] = val_i;
        }
        row_prev = row;
      }

      colind_[val_i] = (IndexT)col;
      values_[val_i] = val;
      val_i++;
    }

    rowptr_[row_i] = val_i;

    // More sanity checks.
    assert(row_i == nrows_);
    assert(val_i == nnz_);

    // reorder();
    // if (nthreads_ > 1)
    //   split_by_bandwidth();
    split_by_nnz(nthreads_);
  }

  // Initialize CSRMatrix from another CSRMatrix matrix (no ownership)
  CSRMatrix(IndexT *rowptr, IndexT *colind, ValueT *values, IndexT nrows,
            IndexT ncols, bool symmetric = false,
            Platform platform = Platform::cpu)
      : platform_(platform), nrows_(nrows), ncols_(ncols),
        symmetric_(symmetric), owns_data_(false) {
    rowptr_ = rowptr;
    colind_ = colind;
    values_ = values;
    nnz_ = rowptr_[nrows];
    // Partitioning
    split_nnz_ = false;
    nthreads_ = get_threads();
    row_split_ = nullptr;
    // Symmetry compression
    cmp_symmetry_ = atomics_ = effective_ranges_ = explicit_conflicts_ = conflict_free_ = false;
    nnz_lower_ = nnz_diag_ = nrows_left_ = nconflicts_ = ncolors_ = 0;
    rowptr_sym_ = colind_sym_ = nullptr;
    values_sym_ = diagonal_ = nullptr;

    // reorder();
    // if (nthreads_ > 1)
    //   split_by_bandwidth();
    split_by_nnz(nthreads_);
  }

  virtual ~CSRMatrix() {
    // If CSRMatrix was initialized using pre-defined arrays, we release
    // ownership.
    if (owns_data_) {
      internal_free(rowptr_, platform_);
      internal_free(colind_, platform_);
      internal_free(values_, platform_);
    } else {
      rowptr_ = nullptr;
      colind_ = nullptr;
      values_ = nullptr;
    }

    internal_free(row_split_, platform_);

    for (size_t i = 0; i < sym_cmp_data_.size(); ++i)
      delete sym_cmp_data_[i];
    sym_cmp_data_.clear();
  }

  virtual int nrows() const override { return nrows_; }
  virtual int ncols() const override { return ncols_; }
  virtual int nnz() const override { return nnz_; }
  virtual bool symmetric() const override { return symmetric_; }

  // FIXME
  virtual size_t size() const override {
    int size = (nrows_ + 1) * sizeof(IndexT); // rowptr
    size += nnz_ * sizeof(IndexT);            // colind
    size += nnz_ * sizeof(ValueT);            // values

    if (split_nnz_)
      size += (nthreads_ + 1) * sizeof(IndexT); // row_split

    if (cmp_symmetry_) {
      size -= nnz_ * sizeof(IndexT);       // colind
      size += nnz_lower_ * sizeof(IndexT); // colind
      size -= nnz_ * sizeof(ValueT);       // values
      size += nnz_lower_ * sizeof(ValueT); // values
      return size;
    }

    return size;
  }

  virtual inline Platform platform() const override { return platform_; }

  virtual bool tune(Kernel k, Tuning t) override {
    if (t == Tuning::None) {
      spmv_fn = boost::bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_vanilla, this, _1, _2);
      return false;
    }

    if (symmetric_) {
#ifdef _LOG_INFO
      cout << "[INFO]: converting CSR format to SSS format..." << endl;
#endif
      compress_symmetry();
      if (nthreads_ == 1) {
	spmv_fn = boost::bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_serial, this, _1, _2);
      } else {
	if (atomics_)
	  spmv_fn = boost::bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_atomics, this, _1, _2);
	else if (effective_ranges_)
	  spmv_fn = boost::bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_effective_ranges, this, _1, _2);
	else if (explicit_conflicts_)
	  spmv_fn = boost::bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_explicit_conflicts, this, _1, _2);
	else if (conflict_free_)
	  spmv_fn = boost::bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free, this, _1, _2);
	else 
	  assert(false);
      }
    } else {
      spmv_fn = boost::bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_split_nnz, this, _1, _2);
    }

    return true;
  }

  virtual void dense_vector_multiply(ValueT *__restrict y,
                                     const ValueT *__restrict x) override {
    spmv_fn(y, x);
  }

  // Export CSR internal representation
  IndexT *rowptr() const { return rowptr_; }
  IndexT *colind() const { return colind_; }
  ValueT *values() const { return values_; }

private:
  Platform platform_;
  int nrows_, ncols_, nnz_, nnz_high_;
  bool symmetric_, owns_data_;
  IndexT *rowptr_, *rowind_, *rowptr_high_;
  IndexT *colind_, *colind_high_;
  ValueT *values_, *values_high_;
  // Internal function pointers
  boost::function<void(ValueT *__restrict, const ValueT *__restrict)> spmv_fn;
  // Reordering
  IndexT *perm_, *inv_perm_;
  bool reordered_;
  // Partitioning
  bool split_nnz_;
  int nthreads_;
  IndexT *row_split_;
  // Symmetry compression
  bool cmp_symmetry_;
  bool atomics_, effective_ranges_, explicit_conflicts_, conflict_free_;
  int nnz_lower_, nnz_diag_, nrows_left_, nconflicts_, ncolors_, nranges_;
  IndexT *rowptr_sym_;
  IndexT *colind_sym_;
  ValueT *values_sym_;
  ValueT *diagonal_;
  IndexT *color_ptr_;
  IndexT *range_ptr_;                // Number of ranges per color
  IndexT *range_start_, *range_end_; // Ranges start/end
  ConflictMap *cnfl_map_;
  ValueT **y_local_;
  vector<SymmetryCompressionData<IndexT, ValueT> *> sym_cmp_data_;
  const int BLK_FACTOR = 1;
  /*
  * Preprocessing routines
  */
  void reorder();
  void split_by_bandwidth();
  void split_by_nrows(int nthreads);
  void split_by_nnz(int nthreads, bool symmetric = false);

  /*
   * Symmetry compression
   */
  void compress_symmetry();
  // Method 1: local vector with effective ranges
  void atomics();
  // Method 2: local vector with effective ranges
  void effective_ranges();
  // Method 3: local vectors indexing
  void explicit_conflicts();
  void count_conflicting_rows();
  // Method 4: conflict-free a priori
  void conflict_free_apriori();
  void count_apriori_conflicts();
  // Method 5: conflict-free a posteriori
  void conflict_free_aposteriori();
  void count_aposteriori_conflicts();
  // Common utilities for methods 4 & 5
  void color(const ColoringGraph &g, ColorMap &color);
  void ordering_heuristic(const ColoringGraph &g, vector<Vertex> &order);
  void natural_vertex_ordering(const ColoringGraph &g, vector<Vertex> &order);
  void natural_round_robin_vertex_ordering(const ColoringGraph &g,
                                           vector<Vertex> &order);
  void smallest_nnz_vertex_ordering(const ColoringGraph &g,
                                    vector<Vertex> &order);
  void smallest_nnz_round_robin_vertex_ordering(const ColoringGraph &g,
                                                vector<Vertex> &order);
  void largest_nnz_vertex_ordering(const ColoringGraph &g,
                                   vector<Vertex> &order);
  void largest_nnz_round_robin_vertex_ordering(const ColoringGraph &g,
                                               vector<Vertex> &order);

  /*
   * Sparse Matrix - Dense Vector Multiplication kernels
   */
  void cpu_mv_vanilla(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_split_nnz(ValueT *__restrict y, const ValueT *__restrict x);

  // Symmetric kernels
  void cpu_mv_sym_serial(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_sym_atomics(ValueT *__restrict y, const ValueT *__restrict x);  
  void cpu_mv_sym_effective_ranges(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_sym_explicit_conflicts(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free_apriori(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free(ValueT *__restrict y, const ValueT *__restrict x);
  void cpu_mv_sym_conflict_free_hyb_bw(ValueT *__restrict y, const ValueT *__restrict x);

  /*
   * Benchmarks for conflict-free SpMV
   */
  void bench_conflict_free_nobarrier(ValueT *__restrict y, const ValueT *__restrict x);
  void bench_conflict_free_nobarrier_noxmiss(ValueT *__restrict y, const ValueT *__restrict x);  
};

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::reorder() {
#ifdef _LOG_INFO
  cout << "[INFO]: reordering matrix using RCM..." << endl;
#endif

  // Construct graph
  typedef boost::adjacency_list<
      boost::vecS, boost::vecS, boost::undirectedS,
      boost::property<boost::vertex_color_t, boost::default_color_type,
                      boost::property<boost::vertex_degree_t, int>>>
      ReorderingGraph;
  ReorderingGraph g(nrows_);
  for (int i = 0; i < nrows_; ++i) {
    for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
      IndexT col = colind_[j];
      if (col != i) {
        add_edge(i, col, g);
      }
    }
  }

#ifdef _LOG_INFO
  size_t ob = bandwidth(g);
  cout << "[INFO]: original bandwidth = " << ob << endl;
#endif

  // Reverse Cuthill Mckee Ordering
  using namespace boost;
  typedef graph_traits<ReorderingGraph>::vertex_descriptor Vertex;
  // typedef graph_traits<ReorderingGraph>::vertices_size_type size_type;
  vector<Vertex> inv_perm(num_vertices(g));
  cuthill_mckee_ordering(g, inv_perm.rbegin(), get(vertex_color, g),
                         make_degree_map(g));

  // Find permutation of original to new ordering
  property_map<ReorderingGraph, vertex_index_t>::type idx_map =
      get(vertex_index, g);
  // FIXME first touch
  perm_ = (IndexT *)internal_alloc(nrows_ * sizeof(IndexT), platform_);
  inv_perm_ = (IndexT *)internal_alloc(nrows_ * sizeof(IndexT), platform_);
  for (size_t i = 0; i != inv_perm.size(); ++i) {
    perm_[idx_map[inv_perm[i]]] = i;
    inv_perm_[i] = inv_perm[i];
  }

#ifdef _LOG_INFO
  size_t fb =
      bandwidth(g, make_iterator_property_map(&perm_[0], idx_map, perm_[0]));
  cout << "[INFO]: final bandwidth = " << fb << endl;
#endif

  // Reorder original matrix
  // First reorder rows
  vector<IndexT> row_nnz(nrows_);
  for (int i = 0; i < nrows_; ++i) {
    row_nnz[perm_[i]] = rowptr_[i + 1] - rowptr_[i];
  }

  IndexT *new_rowptr =
      (IndexT *)internal_alloc((nrows_ + 1) * sizeof(IndexT), platform_);
  // Enforce first touch policy
  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (int i = 1; i <= nrows_; ++i) {
    new_rowptr[i] = row_nnz[i - 1];
  }

  for (int i = 1; i <= nrows_; ++i) {
    new_rowptr[i] += new_rowptr[i - 1];
  }
  assert(new_rowptr[nrows_] == nnz_);

  // Then reorder nonzeros per row
  map<IndexT, ValueT> sorted_row;
  IndexT *new_colind =
      (IndexT *)internal_alloc(nnz_ * sizeof(IndexT), platform_);
  ValueT *new_values =
      (ValueT *)internal_alloc(nnz_ * sizeof(ValueT), platform_);

  // Enforce first touch policy
  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (int i = 0; i < nnz_; i++) {
    new_colind[i] = 0;
    new_values[i] = 0.0;
  }

  for (int i = 0; i < nrows_; ++i) {
    for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
      sorted_row.insert(make_pair(perm_[colind_[j]], values_[j]));
    }

    // Flush row
    auto it = sorted_row.begin();
    for (int j = new_rowptr[perm_[i]]; j < new_rowptr[perm_[i] + 1]; ++j) {
      new_colind[j] = it->first;
      new_values[j] = it->second;
      ++it;
    }

    sorted_row.clear();
  }

  internal_free(rowptr_, platform_);
  internal_free(colind_, platform_);
  internal_free(values_, platform_);
  rowptr_ = new_rowptr;
  colind_ = new_colind;
  values_ = new_values;
  reordered_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::split_by_bandwidth() {
#ifdef _LOG_INFO
  cout << "[INFO]: clustering matrix into low and high bandwidth nonzeros" << endl;
#endif

  vector<IndexT> rowptr_low(nrows_ + 1, 0);
  map<IndexT, IndexT> rowind;
  vector<IndexT> colind_low, colind_high;
  vector<ValueT> values_low, values_high;
  const int THRESHOLD = 20000;

  rowptr_low[0] = 0;
  for (int i = 0; i < nrows_; ++i) {
    for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
      if (abs(colind_[j] - i) < THRESHOLD) {
        rowptr_low[i + 1]++;
        colind_low.push_back(colind_[j]);
        values_low.push_back(values_[j]);
      } else {
        if (rowind.find(i) == rowind.end())
          rowind.insert(pair<IndexT, IndexT>(i, 1));
        else
          rowind[i]++;
        colind_high.push_back(colind_[j]);
        values_high.push_back(values_[j]);
      }
    }
  }

  for (int i = 1; i < nrows_ + 1; ++i) {
    rowptr_low[i] += rowptr_low[i - 1];
  }
  assert(rowptr_low[nrows_] == static_cast<int>(values_low.size()));

  nnz_ = values_low.size();
  move(rowptr_low.begin(), rowptr_low.end(), rowptr_);
  move(colind_low.begin(), colind_low.end(), colind_);
  move(values_low.begin(), values_low.end(), values_);

  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (int i = 0; i <= nrows_; i++) {
    rowptr_[i] = rowptr_low[i];
  }

  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (int i = 0; i < nnz_; i++) {
    colind_[i] = colind_low[i];
    values_[i] = values_low[i];
  }

  nrows_left_ = rowind.size();
  nnz_high_ = values_high.size();
  rowind_ = (IndexT *)internal_alloc(nrows_left_ * sizeof(IndexT), platform_);
  rowptr_high_ =
      (IndexT *)internal_alloc((nrows_left_ + 1) * sizeof(IndexT), platform_);
  colind_high_ =
      (IndexT *)internal_alloc(nnz_high_ * sizeof(IndexT), platform_);
  values_high_ =
      (ValueT *)internal_alloc(nnz_high_ * sizeof(ValueT), platform_);

  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (int i = 0; i <= nrows_left_; i++) {
    rowptr_high_[i] = 0;
  }

  int cnt = 0;
  for (auto &elem : rowind) {
    rowind_[cnt++] = elem.first;
    rowptr_high_[cnt] = elem.second;
  }

  for (int i = 1; i < nrows_left_ + 1; ++i) {
    rowptr_high_[i] += rowptr_high_[i - 1];
  }
  assert(rowptr_high_[nrows_left_] == nnz_high_);

  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (int i = 0; i < nnz_high_; i++) {
    colind_high_[i] = colind_high[i];
    values_high_[i] = values_high[i];
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::split_by_nrows(int nthreads) {
#ifdef _LOG_INFO
  cout << "[INFO]: splitting matrix into " << nthreads << " partitions by rows" << endl;
#endif

  if (!row_split_) {
    row_split_ =
        (IndexT *)internal_alloc((nthreads + 1) * sizeof(IndexT), platform_);
  }

  // Re-init
  memset(row_split_, 0, (nthreads + 1) * sizeof(IndexT));

  // Compute new matrix splits
  int nrows_per_split = nrows_ / nthreads;
  row_split_[0] = 0;
  for (int i = 0; i < nthreads - 1; i++) {
    row_split_[i + 1] += nrows_per_split;
  }

  row_split_[nthreads] = nrows_;
  // split_nrows_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::split_by_nnz(int nthreads, bool symmetric) {
#ifdef _LOG_INFO
  if (symmetric)
    cout << "[INFO]: splitting lower triangular part of matrix into "
         << nthreads << " partitions" << endl;
  else
    cout << "[INFO]: splitting full matrix into " << nthreads << " partitions"
         << endl;
#endif

  if (!row_split_) {
    row_split_ =
        (IndexT *)internal_alloc((nthreads + 1) * sizeof(IndexT), platform_);
  }

  if (nthreads_ == 1) {
    row_split_[0] = 0;
    row_split_[1] = nrows_;
    split_nnz_ = true;
    return;
  }

  // Compute the matrix splits.
  IndexT *row_ptr = (symmetric) ? rowptr_sym_ : rowptr_;
  int nnz_cnt = (symmetric) ? nnz_lower_ : nnz_;
  int nnz_per_split = nnz_cnt / nthreads_;
  int curr_nnz = 0;
  int row_start = 0;
  int split_cnt = 0;
  int i;

  row_split_[0] = row_start;
  for (i = 0; i < nrows_; i++) {
    curr_nnz += row_ptr[i + 1] - row_ptr[i];
    if ((curr_nnz >= nnz_per_split) && ((i + 1) % BLK_FACTOR == 0)) {
      row_start = i + 1;
      curr_nnz = 0;
      ++split_cnt;
      if (split_cnt <= nthreads)
        row_split_[split_cnt] = row_start;
    }
  }

  // Fill the last split with remaining elements
  if (curr_nnz < nnz_per_split && split_cnt <= nthreads) {
    row_split_[++split_cnt] = nrows_;
  }

  // If there are any remaining rows merge them in last partition
  if (split_cnt > nthreads_) {
    row_split_[nthreads_] = nrows_;
  }

  // If there are remaining threads create empty partitions
  for (int i = split_cnt + 1; i <= nthreads; i++) {
    row_split_[i] = nrows_;
  }

  split_nnz_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::atomics() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using atomics" << endl;
#endif

  sym_cmp_data_.resize(nthreads_);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    sym_cmp_data_[tid] = new SymmetryCompressionData<IndexT, ValueT>;
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    int nrows = row_split_[tid + 1] - row_split_[tid];
    int row_offset = row_split_[tid];
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));
    diagonal_ = (ValueT *)internal_alloc(nrows_ * sizeof(ValueT));
    memset(diagonal_, 0, nrows_ * sizeof(IndexT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    size_t nnz_diag = 0;

    data->rowptr_[0] = 0;
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          data->rowptr_[i + 1 - row_offset]++; // FIXME check
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          diagonal_[i] = values_[j];
          data->diagonal_[i - row_offset] = values_[j];
          nnz_diag++;
        }
      }
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_lower_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_lower_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_lower_ * sizeof(ValueT), platform_);

    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = data->rowptr_[i - row_offset];
           j < data->rowptr_[i - row_offset + 1]; ++j) {
        data->colind_[j] = colind_sym[j];
        data->values_[j] = values_sym[j];
      }
    }

    // Cleanup
    colind_sym.clear();
    values_sym.clear();
  }

  for (int tid = 0; tid < nthreads_; ++tid) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    nnz_lower_ += data->nnz_lower_;
    nnz_diag_ += data->nnz_diag_;
  }

  cmp_symmetry_ = true;
  atomics_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::effective_ranges() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using effective ranges of local "
          "vectors"
       << endl;
#endif

  sym_cmp_data_.resize(nthreads_);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    sym_cmp_data_[tid] = new SymmetryCompressionData<IndexT, ValueT>;
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    int nrows = row_split_[tid + 1] - row_split_[tid];
    int row_offset = row_split_[tid];
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));
    diagonal_ = (ValueT *)internal_alloc(nrows_ * sizeof(ValueT));
    memset(diagonal_, 0, nrows_ * sizeof(IndexT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    size_t nnz_diag = 0;

    data->rowptr_[0] = 0;
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          data->rowptr_[i + 1 - row_offset]++; // FIXME check
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          diagonal_[i] = values_[j];
          data->diagonal_[i - row_offset] = values_[j];
          nnz_diag++;
        }
      }
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_lower_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_lower_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_lower_ * sizeof(ValueT), platform_);

    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = data->rowptr_[i - row_offset];
           j < data->rowptr_[i - row_offset + 1]; ++j) {
        data->colind_[j] = colind_sym[j];
        data->values_[j] = values_sym[j];
      }
    }

    if (tid > 0) {
      data->local_vector_ =
          (ValueT *)internal_alloc(row_split_[tid] * sizeof(ValueT), platform_);
      memset(data->local_vector_, 0, row_split_[tid] * sizeof(ValueT));
    }

    // Cleanup
    colind_sym.clear();
    values_sym.clear();
  }

  for (int tid = 0; tid < nthreads_; ++tid) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    nnz_lower_ += data->nnz_lower_;
    nnz_diag_ += data->nnz_diag_;
  }

  cmp_symmetry_ = true;
  effective_ranges_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::count_conflicting_rows() {
  // Sanity check
  assert(cmp_symmetry_);

  int cnfl_total = 0;
  set<IndexT> cnfl[nthreads_ - 1];
  for (int tid = 1; tid < nthreads_; ++tid) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT row_offset = row_split_[tid];
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = data->rowptr_[i - row_offset];
           j < data->rowptr_[i - row_offset + 1]; ++j) {
        if (data->colind_[j] < row_split_[tid])
          cnfl[tid - 1].insert(data->colind_[j]);
      }
    }

    cnfl_total += cnfl[tid - 1].size();
  }

  double cnfl_mean = (double)cnfl_total / (nthreads_ - 1);
  cout << "[INFO]: detected " << cnfl_mean << " mean direct conflicts" << endl;
  cout << "[INFO]: detected " << cnfl_total << " total direct conflicts"
       << endl;
  
  for (int tid = 1; tid < nthreads_; ++tid)
    cnfl[tid - 1].clear();
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::explicit_conflicts() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using explicit conflicts" << endl;
#endif

  y_local_ = (ValueT **)internal_alloc(nthreads_ * sizeof(ValueT *), platform_);
  sym_cmp_data_.resize(nthreads_);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    sym_cmp_data_[tid] = new SymmetryCompressionData<IndexT, ValueT>;
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    int nrows = row_split_[tid + 1] - row_split_[tid];
    int row_offset = row_split_[tid];
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));
    diagonal_ = (ValueT *)internal_alloc(nrows_ * sizeof(ValueT));
    memset(diagonal_, 0, nrows_ * sizeof(IndexT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    size_t nnz_diag = 0;

    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          data->rowptr_[i + 1 - row_offset]++; // FIXME check
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          diagonal_[i] = values_[j];
          data->diagonal_[i - row_offset] = values_[j];
          nnz_diag++;
        }
      }
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_lower_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_lower_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_lower_ * sizeof(ValueT), platform_);

    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = data->rowptr_[i - row_offset];
           j < data->rowptr_[i - row_offset + 1]; ++j) {
        data->colind_[j] = colind_sym[j];
        data->values_[j] = values_sym[j];
      }
    }

    if (tid > 0) {
      data->local_vector_ =
          (ValueT *)internal_alloc(row_split_[tid] * sizeof(ValueT), platform_);
      memset(data->local_vector_, 0, row_split_[tid] * sizeof(ValueT));
      y_local_[tid] = data->local_vector_;
    } else {
      y_local_[tid] = nullptr;
    }

    // Cleanup
    colind_sym.clear();
    values_sym.clear();
  }

  for (int tid = 0; tid < nthreads_; ++tid) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    nnz_lower_ += data->nnz_lower_;
    nnz_diag_ += data->nnz_diag_;
  }

  cmp_symmetry_ = true;

  if (nthreads_ == 1)
    return;

  // Calculate number of conflicting rows per thread and total
  // count_conflicting_rows();

  // Global map of conflicts
  map<IndexT, set<int>> global_map;
  // Conflicting rows per thread
  set<IndexT> thread_map;
  int ncnfls = 0;
  for (int tid = 1; tid < nthreads_; tid++) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT row_offset = row_split_[tid];
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = data->rowptr_[i - row_offset];
           j < data->rowptr_[i + 1 - row_offset]; ++j) {
        IndexT col = data->colind_[j];
        if (col < row_split_[tid]) {
          thread_map.insert(col);
          global_map[col].insert(tid);
        }
      }
    }
    ncnfls += thread_map.size();
    thread_map.clear();
  }

  // Allocate auxiliary map
  cnfl_map_ = new ConflictMap;
  cnfl_map_->length = ncnfls;
  cnfl_map_->cpu = (short *)internal_alloc(ncnfls * sizeof(short), platform_);
  cnfl_map_->pos = (IndexT *)internal_alloc(ncnfls * sizeof(IndexT), platform_);
  int cnt = 0;
  for (auto &elem : global_map) {
    for (auto &cpu : elem.second) {
      cnfl_map_->pos[cnt] = elem.first;
      cnfl_map_->cpu[cnt] = cpu;
      cnt++;
    }
  }
  assert(cnt == ncnfls);

  // Split reduction work among threads so that conflicts to the same row are
  // assigned to the same thread
  int total_count = ncnfls;
  int tid = 0;
  int limit = total_count / nthreads_;
  int tmp_count = 0;
  for (auto &elem : global_map) {
    if (tmp_count <= limit) {
      tmp_count += elem.second.size();
    } else {
      SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
      data->map_end_ = tmp_count;
      // If we have exceeded the number of threads, assigned what is left to
      // last thread
      if (tid == nthreads_ - 1) {
	data->map_end_ = ncnfls;
        break;
      } else {
        total_count -= tmp_count;
        tmp_count = 0;
        limit = total_count / (nthreads_ - tid + 1);
      }
      tid++;
    }
  }

  int start = 0;
  for (int tid = 0; tid < nthreads_; tid++) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    data->map_start_ = start;
    if (tid < nthreads_ - 1)
      data->map_end_ += start;
    start = data->map_end_;
    if (tid == nthreads_ - 1)
      assert(data->map_end_ = ncnfls);
  }

  explicit_conflicts_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::count_apriori_conflicts() {
  // Sanity check
  assert(cmp_symmetry_);

  set<pair<IndexT, IndexT>> cnfl;
  map<IndexT, set<IndexT>> indirect_cnfl;
  for (int i = 0; i < nrows_; ++i) {
    for (int j = rowptr_sym_[i]; j < rowptr_sym_[i + 1]; ++j) {
      cnfl.insert(make_pair(i, colind_sym_[j]));
      indirect_cnfl[colind_sym_[j]].insert(i);
    }
  }

  int no_direct_cnfl = cnfl.size();
  int no_indirect_cnfl = 0;
  // Add indirect conflicts
  for (auto &col : indirect_cnfl) {
    // N * (N-1) / 2
    no_indirect_cnfl += col.second.size() * (col.second.size() - 1) / 2;
    // Create conflicts for every pair of rows in this set
    for (auto &row1 : col.second) {
      for (auto &row2 : col.second) {
        if (row1 != row2) {
          pair<IndexT, IndexT> i_j = make_pair(row1, row2);
          pair<IndexT, IndexT> j_i = make_pair(row2, row1);
          // If these rows are not already connected
          if (cnfl.count(i_j) == 0 && cnfl.count(j_i) == 0)
            cnfl.insert(i_j);
        }
      }
    }
  }

  cout << "[INFO]: detected " << no_direct_cnfl << " direct conflicts" << endl;
  cout << "[INFO]: detected " << no_indirect_cnfl << " indirect conflicts"
       << endl;
  // The number of edges in the graph will be the union of direct and indirect
  // conflicts
  cout << "[INFO]: the a priori conflict graph will contain " << cnfl.size()
       << " edges" << endl;
  
  cnfl.clear();
  indirect_cnfl.clear();
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::count_aposteriori_conflicts() {
  // Sanity check
  assert(cmp_symmetry_);

  set<pair<IndexT, IndexT>> cnfl;
  map<IndexT, vector<pair<IndexT, IndexT>>> indirect_cnfl;
  for (int tid = 0; tid < nthreads_; ++tid) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT row_offset = row_split_[tid];
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = data->rowptr_[i - row_offset];
           j < data->rowptr_[i - row_offset + 1]; ++j) {
        if (data->colind_[j] < row_split_[tid])
          cnfl.insert(make_pair(i, data->colind_[j]));

        indirect_cnfl[data->colind_[j]].push_back(make_pair(i, tid));
      }
    }
  }

  int no_direct_cnfl = cnfl.size();
  int no_indirect_cnfl = 0;

  for (auto &col : indirect_cnfl) {
    for (auto &row1 : col.second) {
      for (auto &row2 : col.second) {
        if (row1.first != row2.first && row1.second != row2.second) {
          pair<IndexT, IndexT> i_j = make_pair(row1.first, row2.first);
          cnfl.insert(i_j);
          no_indirect_cnfl++;
        }
      }
    }
  }

  cout << "[INFO]: detected " << no_direct_cnfl << " direct conflicts" << endl;
  cout << "[INFO]: detected " << no_indirect_cnfl << " indirect conflicts"
       << endl;
  // The number of edges in the graph will be the union of direct and indirect
  // conflicts
  cout << "[INFO]: the a posteriori conflict graph will contain " << cnfl.size()
       << " edges" << endl;
  
  cnfl.clear();
  indirect_cnfl.clear();
}

// FIXME: NUMA?
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::conflict_free_apriori() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using a priori conflict-free SpMV"
       << endl;
#endif

  rowptr_sym_ = (IndexT *)internal_alloc((nrows_ + 1) * sizeof(IndexT));
  memset(rowptr_sym_, 0, (nrows_ + 1) * sizeof(IndexT));
  diagonal_ = (ValueT *)internal_alloc(nrows_ * sizeof(ValueT));
  memset(diagonal_, 0, nrows_ * sizeof(ValueT));

  vector<IndexT> colind_sym;
  vector<ValueT> values_sym;
  nnz_diag_ = 0;
  rowptr_sym_[0] = 0;
  for (int tid = 0; tid < nthreads_; ++tid) {
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          rowptr_sym_[i + 1]++;
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          diagonal_[i] = values_[j];
          nnz_diag_++;
        }
      }
    }
  }

  for (int i = 1; i <= nrows_; ++i) {
    rowptr_sym_[i] += rowptr_sym_[i - 1];
  }

  assert(rowptr_sym_[nrows_] == static_cast<int>(values_sym.size()));
  nnz_lower_ = values_sym.size();
  colind_sym_ =
      (IndexT *)internal_alloc(nnz_lower_ * sizeof(IndexT), platform_);
  values_sym_ =
      (ValueT *)internal_alloc(nnz_lower_ * sizeof(ValueT), platform_);

  for (int j = 0; j < nnz_lower_; ++j) {
    colind_sym_[j] = colind_sym[j];
    values_sym_[j] = values_sym[j];
  }

  cmp_symmetry_ = true;

  // Cleanup
  colind_sym.clear();
  values_sym.clear();

  if (nthreads_ == 1)
    return;

  // Find number of conflicts
  // count_apriori_conflicts();

  // Create conflict graph
  typedef boost::graph_traits<ColoringGraph>::vertex_iterator VertexIterator;
  ColoringGraph g(nrows_);
  map<IndexT, set<IndexT>> indirect_cnfl;

  // First add direct conflicts
  for (int i = 0; i < nrows_; ++i) {
    for (int j = rowptr_sym_[i]; j < rowptr_sym_[i + 1]; ++j) {
      add_edge(i, colind_sym_[j], g);
      indirect_cnfl[colind_sym_[j]].insert(i);
    }
  }

  // Now add indirect conflicts.
  // Indirect conflicts occur when two rows that belong to different threads
  // have nonzero elements in the same column.
  for (auto &col : indirect_cnfl) {
    for (auto &row1 : col.second) {
      for (auto &row2 : col.second) {
        if (row1 != row2) {
          add_edge(row1, row2, g);
        }
      }
    }
  }
  indirect_cnfl.clear();

  // #ifdef _LOG_INFO
  // typedef boost::graph_traits<ColoringGraph>::degree_size_type Degree;
  //   // Find maximum vertex degree.
  //   VertexIterator ui, ui_end;
  //   Degree dummy, maximum_degree = 0;
  //   for (boost::tie(ui, ui_end) = vertices(g); ui != ui_end; ++ui) {
  //     dummy = boost::degree(*ui, g);
  //     if (dummy > maximum_degree) {
  //       maximum_degree = dummy;
  //     }
  //   }
  //   cout << "[INFO]: maximum vertex degree is " << maximum_degree << endl;
  // #endif

  // Run graph coloring
  vector<vertices_size_type> color_vec(num_vertices(g));
  ColorMap color_map(&color_vec.front(), get(boost::vertex_index, g));
  color(g, color_map);

  // Find row indices per color
  vector<IndexT> rowind[ncolors_];
  VertexIterator v, v_end;
  boost::property_map<ColoringGraph, boost::vertex_index_t>::type vertIndx =
      boost::get(boost::vertex_index, g);
  for (boost::tie(v, v_end) = vertices(g); v != v_end; v++) {
    rowind[color_map[*v]].push_back(vertIndx[*v]);
  }

  // Allocate auxiliary arrays
  color_ptr_ = (IndexT *)internal_alloc((ncolors_ + 1) * sizeof(IndexT));
  memset(color_ptr_, 0, (ncolors_ + 1) * sizeof(IndexT));
  for (int c = 1; c <= ncolors_; ++c) {
    color_ptr_[c] += color_ptr_[c - 1] + rowind[c - 1].size();
  }
  assert(color_ptr_[ncolors_] == nrows_);

  rowind_ = (IndexT *)internal_alloc(nrows_ * sizeof(IndexT));
  int cnt = 0;
  for (int c = 0; c < ncolors_; ++c) {
    sort(rowind[c].begin(), rowind[c].end());
    for (size_t i = 0; i < rowind[c].size(); ++i) {
      rowind_[cnt++] = rowind[c][i];
    }
  }

  conflict_free_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::conflict_free_aposteriori() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using a posteriori conflict-free "
          "SpMV"
       << endl;
#endif

  ColoringGraph g(ceil(nrows_ / (double)BLK_FACTOR));
#ifdef _LOG_INFO
  double tstart, tstop;
  tstart = omp_get_wtime();
#endif
  sym_cmp_data_.resize(nthreads_);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    sym_cmp_data_[tid] = new SymmetryCompressionData<IndexT, ValueT>;
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    int nrows = row_split_[tid + 1] - row_split_[tid];
    int row_offset = row_split_[tid];
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));
    memset(data->diagonal_, 0, nrows * sizeof(IndexT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    size_t nnz_diag = 0;

    data->rowptr_[0] = 0;
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        IndexT col = colind_[j];
        if (col < i) {
          data->rowptr_[i + 1 - row_offset]++;
          colind_sym.push_back(col);
          values_sym.push_back(values_[j]);
        } else if (col == i) {
          data->diagonal_[i - row_offset] = values_[j];
          nnz_diag++;
        }
      }
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_lower_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_lower_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_lower_ * sizeof(ValueT), platform_);

    std::move(colind_sym.begin(), colind_sym.end(), data->colind_);
    std::move(values_sym.begin(), values_sym.end(), data->values_);

    // Cleanup
    colind_sym.clear();
    values_sym.clear();
  }

  for (int tid = 0; tid < nthreads_; ++tid) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    nnz_lower_ += data->nnz_lower_;
  }

  cmp_symmetry_ = true;

  if (nthreads_ == 1)
    return;

  // Find number of conflicts
  // count_aposteriori_conflicts();

  vector<vector<pair<int, int>>> indirect_cnfl(nrows_);
  for (int t = 0; t < nthreads_; t++) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[t];
    IndexT row_offset = row_split_[t];
    for (int i = row_split_[t]; i < row_split_[t + 1]; i++) {
      for (int j = data->rowptr_[i - row_offset];
           j < data->rowptr_[i + 1 - row_offset]; j++) {
        IndexT col = data->colind_[j];
        // If this nonzero is in the lower triangular part and has a direct
        // conflict with another thread
        if (col < row_offset) {
          add_edge(i / BLK_FACTOR, col / BLK_FACTOR, g);
        }

        for (auto &row : indirect_cnfl[col])
          if (row.second != t)
            add_edge(row.first / BLK_FACTOR, i / BLK_FACTOR, g);
        indirect_cnfl[col].emplace_back(make_pair(i, t));
      }
    }
  }

  // Run graph coloring
  vector<vertices_size_type> color_vec(num_vertices(g));
  ColorMap color_map(&color_vec.front(), get(boost::vertex_index, g));
  color(g, color_map);

  // Find row sets per thread per color
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];

    vector<vector<IndexT>> rowind(ncolors_);

    // Find active row indices per color
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; i++) {
        rowind[color_map[i / BLK_FACTOR]].push_back(i);
    }

    // Detect ranges of consecutive rows
    vector<IndexT> row_start[ncolors_];
    vector<IndexT> row_end[ncolors_];
    IndexT row, row_prev;
    int nranges = 0;
    for (int c = 0; c < ncolors_; c++) {
      // assert(static_cast<int>(rowind[c].size()) <=
      //        row_split_[tid + 1] - row_split_[tid] + 1);
      if (rowind[c].size() > 0) {
        row_prev = rowind[c][0];
        row_start[c].push_back(row_prev);
        for (auto it = rowind[c].begin(); it != rowind[c].end(); ++it) {
          row = *it;
          if (row - row_prev > 1) {
            row_end[c].push_back(row_prev);
            row_start[c].push_back(row);
          }

          row_prev = row;
        }

        // Finalize row_end
        row_end[c].push_back(row);
      }

      nranges += row_start[c].size();
    }

    rowind.clear();

    // Allocate auxiliary arrays
    data->ncolors_ = ncolors_;
    data->nranges_ = nranges;
    data->range_ptr_ =
        (IndexT *)internal_alloc((ncolors_ + 1) * sizeof(IndexT));
    data->range_start_ = (IndexT *)internal_alloc(nranges * sizeof(IndexT));
    data->range_end_ = (IndexT *)internal_alloc(nranges * sizeof(IndexT));
    int cnt = 0;
    int row_offset = row_split_[tid];
    data->range_ptr_[0] = 0;
    for (int c = 0; c < ncolors_; c++) {
      data->range_ptr_[c + 1] = data->range_ptr_[c] + row_start[c].size();
      if (row_start[c].size() > 0) {
        for (int i = 0; i < static_cast<int>(row_start[c].size()); ++i) {
          data->range_start_[cnt] = row_start[c][i] - row_offset;
          data->range_end_[cnt] = row_end[c][i] - row_offset;
          cnt++;
        }
      }
    }
    assert(cnt == nranges);
  }

#ifdef _LOG_INFO
  tstop = omp_get_wtime();
  cout << "[INFO]: conversion time: " << tstop - tstart << endl;
#endif
  conflict_free_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::compress_symmetry() {
  if (!symmetric_) {
    return;
  }

  // atomics();
  // effective_ranges();
  // explicit_conflicts();
  // conflict_free_apriori();
  conflict_free_aposteriori();
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::natural_vertex_ordering(
    const ColoringGraph &g, std::vector<Vertex> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying N vertex ordering..." << endl;
#endif
  
  typedef typename boost::graph_traits<ColoringGraph>::vertex_iterator
      VertexIterator;
  std::pair<VertexIterator, VertexIterator> v = vertices(g);
  while (v.first != v.second)
    order.push_back(*v.first++);
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::natural_round_robin_vertex_ordering(
    const ColoringGraph &g, std::vector<Vertex> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying N-RR vertex ordering..." << endl;
#endif
  
  int cnt = 0, t_cnt = 0;
  while ((unsigned int)cnt < num_vertices(g)) {
    for (int t = 0; t < nthreads_; t++) {
      if (row_split_[t] + t_cnt < row_split_[t + 1]) {
        assert(((row_split_[t] + t_cnt) / BLK_FACTOR) < nrows_);
        order.push_back(vertex((row_split_[t] + t_cnt) / BLK_FACTOR, g));
        cnt++;
      }
    }

    t_cnt += BLK_FACTOR;
  }

  assert(order.size() == num_vertices(g));
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::smallest_nnz_vertex_ordering(
    const ColoringGraph &g, std::vector<Vertex> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying SNNZ vertex ordering..." << endl;
#endif
  
  // Sort rows by increasing number of nonzeros
  std::multimap<size_t, IndexT> row_nnz;
  for (int t = 0; t < nthreads_; ++t) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[t];
    IndexT row_offset = row_split_[t];
    for (int i = 0; i < data->nrows_; ++i) {
      int nnz = data->rowptr_[i + 1] - data->rowptr_[i];
      row_nnz.insert(std::pair<size_t, IndexT>(nnz, i + row_offset));
    }
  }

  for (auto it = row_nnz.begin(); it != row_nnz.end(); ++it) {
    order.push_back(it->second);
  }

  assert(order.size() == num_vertices(g));
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::smallest_nnz_round_robin_vertex_ordering(
    const ColoringGraph &g, std::vector<Vertex> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying SNNZ-RR vertex ordering..." << endl;
#endif
  
  // Sort rows by number of nonzeros per thread
  std::multimap<size_t, IndexT> row_nnz[nthreads_];
  for (int t = 0; t < nthreads_; ++t) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[t];
    IndexT row_offset = row_split_[t];
    for (int i = 0; i < data->nrows_; ++i) {
      int nnz = data->rowptr_[i + 1] - data->rowptr_[i];
      row_nnz[t].insert(std::pair<size_t, IndexT>(nnz, i + row_offset));
    }
  }

  typename std::multimap<size_t, IndexT>::iterator it[nthreads_];
  for (int t = 0; t < nthreads_; t++) {
    it[t] = row_nnz[t].begin();
  }

  int cnt = 0;
  while (cnt < nrows_) {
    for (int t = 0; t < nthreads_; t++) {
      if (it[t] != row_nnz[t].end()) {
        order.push_back(it[t]->second);
        it[t]++;
        cnt++;
      }
    }
  }

  assert(order.size() == num_vertices(g));
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::largest_nnz_vertex_ordering(
    const ColoringGraph &g, std::vector<Vertex> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying LNNZ vertex ordering..." << endl;
#endif
  
  // Sort rows by decreasing number of nonzeros
  std::multimap<size_t, IndexT> row_nnz;
  for (int t = 0; t < nthreads_; ++t) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[t];
    IndexT row_offset = row_split_[t];
    for (int i = 0; i < data->nrows_; ++i) {
      int nnz = data->rowptr_[i + 1] - data->rowptr_[i];
      row_nnz.insert(std::pair<size_t, IndexT>(nnz, i + row_offset));
    }
  }

  for (auto it = row_nnz.rbegin(); it != row_nnz.rend(); ++it) {
    order.push_back(it->second);
  }

  assert(order.size() == num_vertices(g));
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::largest_nnz_round_robin_vertex_ordering(
    const ColoringGraph &g, std::vector<Vertex> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying LNNZ-RR vertex ordering..." << endl;
#endif
  
  // Sort rows by number of nonzeros per thread
  std::multimap<size_t, IndexT> row_nnz[nthreads_];
  for (int t = 0; t < nthreads_; ++t) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[t];
    IndexT row_offset = row_split_[t];
    for (int i = 0; i < data->nrows_; ++i) {
      int nnz = data->rowptr_[i + 1] - data->rowptr_[i];
      row_nnz[t].insert(std::pair<size_t, IndexT>(nnz, i + row_offset));
    }
  }

  typename std::multimap<size_t, IndexT>::reverse_iterator it[nthreads_];
  for (int t = 0; t < nthreads_; t++) {
    it[t] = row_nnz[t].rbegin();
  }

  int cnt = 0;
  while (cnt < nrows_) {
    for (int t = 0; t < nthreads_; t++) {
      if (it[t] != row_nnz[t].rend()) {
        order.push_back(it[t]->second);
        it[t]++;
        cnt++;
      }
    }
  }

  assert(static_cast<int>(order.size()) == nrows_);
}

// N:            Colors vertices in the order they appear in the graph
//               representation.
// N-RR:         Colors vertices in a round-robin fashion among threads
//               but in the order they appear in the graph representation.
// SNNZ:         Colors vertices in increasing row size.
// SNNZ-RR:      Colors vertices in a round-robin fashion among threads
//               but in increasing row size order.
// LNNZ:         Colors vertices in decreasing row size.
// LNNZ-RR:      Colors vertices in a round-robin fashion among threads
//               but in decreasing row size order.
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::ordering_heuristic(const ColoringGraph &g,
                                                   vector<Vertex> &order) {
  order.reserve(num_vertices(g));

// #ifdef _LOG_INFO
//   cout << "[INFO]: applying smallest last vertex ordering..." << endl;
// #endif
  // order = smallest_last_vertex_ordering(g);
  // natural_vertex_ordering(g, order);
  // natural_round_robin_vertex_ordering(g, order);
  // smallest_nnz_vertex_ordering(g, order);
  // smallest_nnz_round_robin_vertex_ordering(g, order);
  // largest_nnz_vertex_ordering(g, order);
  largest_nnz_round_robin_vertex_ordering(g, order);
}

// Assumes matrix rows have been assigned to threads
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::color(const ColoringGraph &g, ColorMap &color) {
  assert(symmetric_ && cmp_symmetry_);

#ifdef _LOG_INFO
  cout << "[INFO]: applying graph coloring to detect conflict-free submatrices"
       << endl;
#endif

  // Modify vertex ordering to improve coloring
  vector<Vertex> order;
  ordering_heuristic(g, order);

#ifdef _LOG_INFO
  float tstart = omp_get_wtime();
#endif
  // ncolors_ = sequential_vertex_coloring(g, color);
  ncolors_ = sequential_vertex_coloring
    (g, make_iterator_property_map(order.begin(),
    boost::identity_property_map(),
    boost::graph_traits<ColoringGraph>::null_vertex()), color);
#ifdef _LOG_INFO  
  float tstop = omp_get_wtime();
  cout << "[INFO]: graph coloring: " << tstop - tstart << endl;
  cout << "[INFO]: using " << ncolors_ << " colors" << endl;
#endif
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_vanilla(ValueT *__restrict y,
                                               const ValueT *__restrict x) {
  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (int i = 0; i < nrows_; ++i) {
    ValueT y_tmp = 0.0;

    PRAGMA_IVDEP
    for (IndexT j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
      y_tmp += values_[j] * x[colind_[j]];
    }

    y[i] = y_tmp;
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_split_nnz(ValueT *__restrict y,
                                                 const ValueT *__restrict x) {
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    for (IndexT i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      register ValueT y_tmp = 0;

      PRAGMA_IVDEP
      for (IndexT j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
#if defined(_PREFETCH) && defined(_INTEL_COMPILER)
        /* T0: prefetch into L1, temporal with respect to all level caches */
        _mm_prefetch((const char *)&x[colind_[j + 16]], _MM_HINT_T2);
#endif
        y_tmp += values_[j] * x[colind_[j]];
      }

      /* Reduction on y */
      y[i] = y_tmp;
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_serial(ValueT *__restrict y,
                                                  const ValueT *__restrict x) {

  SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[0];
  IndexT *rowptr = data->rowptr_;
  IndexT *colind = data->colind_;
  ValueT *values = data->values_;
  ValueT *diagonal = data->diagonal_;

  for (int i = 0; i < nrows_; ++i) {
    ValueT y_tmp = diagonal[i] * x[i];

    PRAGMA_IVDEP
    for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      IndexT col = colind[j];
      ValueT val = values[j];
      y_tmp += val * x[col];
      y[col] += val * x[i];
    }

    /* Reduction on y */
    y[i] = y_tmp;
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_atomics(
    ValueT *__restrict y, const ValueT *__restrict x) {

  // Local vectors phase
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier
    
    for (int i = 0; i < data->nrows_; ++i) {
      ValueT y_tmp = 0;

      PRAGMA_IVDEP
      for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        IndexT col = colind[j];
        ValueT val = values[j];
        y_tmp += val * x[col];
	#pragma omp atomic
	y[col] += val * x[i + row_offset];
      }

      /* Reduction on y */
      y[i + row_offset] += y_tmp;
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_effective_ranges(
    ValueT *__restrict y, const ValueT *__restrict x) {

  // Local vectors phase
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    ValueT *y_local = data->local_vector_;
    if (tid == 0)
      y_local = y;
    memset(y_local, 0.0, row_split_[tid] * sizeof(ValueT));

    for (int i = 0; i < data->nrows_; ++i) {
      ValueT y_tmp = diagonal[i] * x[i + row_offset];

      PRAGMA_IVDEP
      for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        IndexT col = colind[j];
        ValueT val = values[j];
        y_tmp += val * x[col];
        if (col < row_split_[tid])
          y_local[col] += val * x[i + row_offset];
        else
          y[col] += val * x[i + row_offset];
      }

      /* Reduction on y */
      y[i + row_offset] = y_tmp;
    }
  }

  // Reduction of conflicts phase
  for (int tid = 1; tid < nthreads_; ++tid) {
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    ValueT *y_local = data->local_vector_;
    #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
    for (IndexT i = 0; i < row_split_[tid]; ++i) {
      y[i] += y_local[i];
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_explicit_conflicts(
    ValueT *__restrict y, const ValueT *__restrict x) {

  // Local vectors phase
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    ValueT *y_local = data->local_vector_;
    if (tid == 0) {
      y_local = y;
    } else {
      memset(y_local, 0.0, row_split_[tid] * sizeof(ValueT));
    }
    
    for (int i = 0; i < data->nrows_; ++i) {
      ValueT y_tmp = diagonal[i] * x[i + row_offset];

      PRAGMA_IVDEP
      for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        IndexT col = colind[j];
        ValueT val = values[j];
        y_tmp += val * x[col];
        if (col < row_split_[tid])
          y_local[col] += val * x[i + row_offset];
        else
          y[col] += val * x[i + row_offset];
      }

      /* Reduction on y */
      y[i + row_offset] = y_tmp;
    }

    #pragma omp barrier
    for (int i = data->map_start_; i < data->map_end_; ++i)
      y[cnfl_map_->pos[i]] += y_local_[cnfl_map_->cpu[i]][cnfl_map_->pos[i]];
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_apriori(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    for (int c = 0; c < ncolors_; ++c) {
      #pragma omp for schedule(runtime)
      for (int i = color_ptr_[c]; i < color_ptr_[c + 1]; ++i) {
        register IndexT row = rowind_[i];
        register ValueT y_tmp = diagonal_[row] * x[row];

        PRAGMA_IVDEP
        for (IndexT j = rowptr_sym_[row]; j < rowptr_sym_[row + 1]; ++j) {
          IndexT col = colind_sym_[j];
          ValueT val = values_sym_[j];
          y_tmp += val * x[col];
          y[col] += val * x[row];
        }

        /* Reduction on y */
        y[row] += y_tmp;
      }
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          register ValueT y_tmp = 0;

          PRAGMA_IVDEP
          for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IndexT col = colind[j];
            ValueT val = values[j];
            y_tmp += val * x[col];
            y[col] += val * x[i + row_offset];
          }

          /* Reduction on y */
          y[i + row_offset] += y_tmp;
        }
      }

      #pragma omp barrier
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_hyb_bw(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          register ValueT y_tmp = 0;

          PRAGMA_IVDEP
          for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IndexT col = colind[j];
            ValueT val = values[j];
            y_tmp += val * x[col];
            y[col] += val * x[i + row_offset];
          }

          /* Reduction on y */
          y[i + row_offset] += y_tmp;
        }
      }

      #pragma omp barrier
    }
  }

  #pragma omp parallel for schedule(runtime) num_threads(nthreads_)
  for (IndexT i = 0; i < nrows_left_; ++i) {
    IndexT row = rowind_[i];
    register ValueT y_tmp = 0;

    PRAGMA_IVDEP
    for (IndexT j = rowptr_high_[i]; j < rowptr_high_[i + 1]; ++j) {
      y_tmp += values_high_[j] * x[colind_high_[j]];
    }

    /* Reduction on y */
    y[row] += y_tmp;
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::bench_conflict_free_nobarrier(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          register ValueT y_tmp = 0;

          PRAGMA_IVDEP
          for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IndexT col = colind[j];
            ValueT val = values[j];
            y_tmp += val * x[col];
            y[col] += val * x[i + row_offset];
          }

          /* Reduction on y */
          y[i + row_offset] += y_tmp;
        }
      }
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::bench_conflict_free_nobarrier_noxmiss(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymmetryCompressionData<IndexT, ValueT> *data = sym_cmp_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          register ValueT y_tmp = 0;

          PRAGMA_IVDEP
          for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IndexT col = colind[j];
            ValueT val = values[j];
            y_tmp += val * x[i + row_offset];
            y[col] += val * x[i + row_offset];
          }

          /* Reduction on y */
          y[i + row_offset] += y_tmp;
        }
      }
    }
  }
}
  
} // end of namespace sparse
} // end of namespace matrix
