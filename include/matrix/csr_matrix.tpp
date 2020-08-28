#ifndef CSR_MATRIX_TPP
#define CSR_MATRIX_TPP

namespace cfs {
namespace matrix {
namespace sparse {

template <typename IndexT, typename ValueT>
CSRMatrix<IndexT, ValueT>::CSRMatrix(const string &filename, Platform platform,
                                     bool symmetric, bool hybrid)
    : platform_(platform), owns_data_(true) {
  MMF<IndexT, ValueT> mmf(filename);
  symmetric_ = mmf.IsSymmetric();
  if (!symmetric) {
    symmetric_ = false;
#ifdef _LOG_INFO
    cout << "[INFO]: using CSR format to store the sparse matrix..." << endl;
#endif
  }
#ifdef _LOG_INFO
  if (symmetric) {
    if (symmetric_ != symmetric) {
      cout << "[INFO]: matrix is not symmetric!" << endl;
      cout << "[INFO]: rolling back to CSR format..." << endl;
    } else {
      if (hybrid) {
        cout << "[INFO]: using HYB format to store the sparse matrix..."
             << endl;
      } else {
        cout << "[INFO]: using SSS format to store the sparse matrix..."
             << endl;
      }
    }
  }
#endif
  nrows_ = mmf.GetNrRows();
  ncols_ = mmf.GetNrCols();
  nnz_ = mmf.GetNrNonzeros();
  rowptr_ = (IndexT *)internal_alloc((nrows_ + 1) * sizeof(IndexT), platform_);
  colind_ = (IndexT *)internal_alloc(nnz_ * sizeof(IndexT), platform_);
  values_ = (ValueT *)internal_alloc(nnz_ * sizeof(ValueT), platform_);
  // Hybrid
  hybrid_ = hybrid;
  split_by_bw_ = false;
  nnz_lbw_ = nnz_hbw_ = 0;
  rowptr_h_ = colind_h_ = nullptr;
  values_h_ = nullptr;
  // Partitioning
  part_by_nrows_ = part_by_nnz_ = part_by_ncnfls_ = false;
  nthreads_ = get_num_threads();
  row_split_ = row_part_ = nullptr;
  // Symmetry compression
  cmp_symmetry_ = false;
  atomics_ = effective_ranges_ = local_vectors_indexing_ = false;
  conflict_free_apriori_ = conflict_free_aposteriori_ = false;
  nnz_low_ = nnz_diag_ = 0;
  ncnfls_ = ncolors_ = nranges_ = 0;
  cnfl_map_ = nullptr;

  // Enforce first touch policy
  #pragma omp parallel num_threads(nthreads_)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < nrows_ + 1; i++) {
      rowptr_[i] = 0;
    }
    #pragma omp for schedule(static)
    for (int i = 0; i < nnz_; i++) {
      colind_[i] = 0;
      values_[i] = 0.0;
    }
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

  if (nthreads_ == 1)
    hybrid_ = false;
}

template <typename IndexT, typename ValueT>
CSRMatrix<IndexT, ValueT>::CSRMatrix(IndexT *rowptr, IndexT *colind,
                                     ValueT *values, IndexT nrows, IndexT ncols,
                                     bool symmetric, bool hybrid,
                                     Platform platform)
    : platform_(platform), nrows_(nrows), ncols_(ncols), symmetric_(symmetric),
      owns_data_(false) {
  rowptr_ = rowptr;
  colind_ = colind;
  values_ = values;
  nnz_ = rowptr_[nrows];
  // Hybrid
  hybrid_ = hybrid;
  split_by_bw_ = false;
  nnz_lbw_ = nnz_hbw_ = 0;
  rowptr_h_ = colind_h_ = nullptr;
  values_h_ = nullptr;
  // Partitioning
  part_by_nrows_ = part_by_nnz_ = part_by_ncnfls_ = false;
  nthreads_ = get_num_threads();
  row_split_ = row_part_ = nullptr;
  // Symmetry compression
  cmp_symmetry_ = false;
  atomics_ = effective_ranges_ = local_vectors_indexing_ = false;
  conflict_free_apriori_ = conflict_free_aposteriori_ = false;
  nnz_low_ = nnz_diag_ = 0;
  ncnfls_ = ncolors_ = nranges_ = 0;
  cnfl_map_ = nullptr;

  if (nthreads_ == 1)
    hybrid_ = false;
}

template <typename IndexT, typename ValueT>
CSRMatrix<IndexT, ValueT>::~CSRMatrix() {
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

  // Hybrid
  if (hybrid_) {
    internal_free(rowptr_h_, platform_);
    internal_free(colind_h_, platform_);
    internal_free(values_h_, platform_);
  }

  // Partitioning
  internal_free(row_split_, platform_);
  internal_free(row_part_, platform_);

  // Symmetry compression
  if (cmp_symmetry_) {
    if (local_vectors_indexing_) {
      internal_free(cnfl_map_->cpu);
      internal_free(cnfl_map_->pos);
      delete cnfl_map_;
    }

    if (conflict_free_apriori_) {
      delete sym_thread_data_[0];
    } else {
      for (int i = 0; i < nthreads_; ++i)
        delete sym_thread_data_[i];
    }
    internal_free(sym_thread_data_, platform_);
  }
}

// FIXME
template <typename IndexT, typename ValueT>
size_t CSRMatrix<IndexT, ValueT>::size() const {
  size_t size = 0;
  if (cmp_symmetry_) {
    size += (nrows_ + 1 * nthreads_) * sizeof(IndexT); // rowptr
    size += nnz_low_ * sizeof(IndexT);                 // colind
    size += nnz_low_ * sizeof(ValueT);                 // values
    size += nnz_diag_ * sizeof(ValueT);                // diagonal

    if (local_vectors_indexing_) {
      size += 2 * nthreads_ * sizeof(IndexT);     // map start/end
      size += cnfl_map_->length * sizeof(short);  // map cpu
      size += cnfl_map_->length * sizeof(IndexT); // map pos
    } else if (conflict_free_apriori_) {
      size += (ncolors_ + 1) * sizeof(IndexT); // col_ptr
      size += nrows_ * sizeof(IndexT);         // rowind_sym
    } else if (conflict_free_aposteriori_) {
      size += (ncolors_ + 1) * sizeof(IndexT); // range_ptr
      size += 2 * nranges_ * sizeof(IndexT);   // range_start/end
    }

    if (hybrid_) {
      // FIXME
      //        size += (nrows_left_ + 1) * sizeof(IndexT); // rowptr_high
      size += nnz_hbw_ * sizeof(IndexT); // colind_high
      size += nnz_hbw_ * sizeof(ValueT); // values_high
    }

    return size;
  }

  size += (nrows_ + 1) * sizeof(IndexT); // rowptr
  size += nnz_ * sizeof(IndexT);         // colind
  size += nnz_ * sizeof(ValueT);         // values
  if (part_by_nnz_)
    size += (nthreads_ + 1) * sizeof(IndexT); // row_split

  return size;
}

template <typename IndexT, typename ValueT>
bool CSRMatrix<IndexT, ValueT>::tune(Kernel k, Tuning t) {
  using placeholders::_1;
  using placeholders::_2;

  // FIXME
  // Matrix decomposition can either precede or proceed row partitioning
  // if (nthreads_ > 1 && hybrid_)
  //   split_by_bandwidth();

  // Partition work
  if (nthreads_ > 1) {
    if (symmetric_) {
#if defined(_METIS) || defined(_KAHIP)
      partition_by_conflicts(nthreads_);
#else
      // partition_by_nnz(nthreads_);
      partition_by_nrows(nthreads_);
#endif
    } else {
      if (t == Tuning::Aggressive) {
        partition_by_nnz(nthreads_);
      } else {
        partition_by_nrows(nthreads_);
      }
    }
  }

  // Decompose matrix
  if (nthreads_ > 1 && hybrid_) {
    split_by_bandwidth();
  }

  if (symmetric_) {
    compress_symmetry();
    if (nthreads_ == 1) {
      spmv_fn =
          bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_serial, this, _1, _2);
    } else {
      if (atomics_)
        spmv_fn =
            bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_atomics, this, _1, _2);
      else if (effective_ranges_)
        spmv_fn = bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_effective_ranges,
                       this, _1, _2);
      else if (local_vectors_indexing_)
        spmv_fn =
            bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_local_vectors_indexing,
                 this, _1, _2);
      else if (conflict_free_apriori_)
        spmv_fn =
            bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_apriori,
                 this, _1, _2);
      else if (conflict_free_aposteriori_ && hybrid_ && part_by_ncnfls_)
        spmv_fn =
            bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_hyb_v1,
                 this, _1, _2);
      else if (conflict_free_aposteriori_ && hybrid_ && part_by_nnz_)
        spmv_fn =
            bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_hyb_v2,
                 this, _1, _2);
      else if (conflict_free_aposteriori_ && !hybrid_ && part_by_ncnfls_)
        spmv_fn = bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_v1,
                       this, _1, _2);
      else if (conflict_free_aposteriori_ && !hybrid_ &&
               (part_by_nnz_ || part_by_nrows_))
        spmv_fn = bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_v2,
                       this, _1, _2);
      else
        assert(false);
    }
  } else {
    if (nthreads_ == 1) {
      spmv_fn = bind(&CSRMatrix<IndexT, ValueT>::cpu_mv_serial, this, _1, _2);
    } else {
      spmv_fn = bind(&CSRMatrix<IndexT, ValueT>::cpu_mv, this, _1, _2);
    }
  }

  return true;
}

// This can either precede or proceed the row assignment
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::split_by_bandwidth() {
#ifdef _LOG_INFO
  cout << "[INFO]: clustering matrix into low and high bandwidth nonzeros"
       << endl;
#endif

  vector<IndexT> rowptr_low(nrows_ + 1, 0);
  vector<IndexT> rowptr_high(nrows_ + 1, 0);
  vector<IndexT> colind_low, colind_high;
  vector<ValueT> values_low, values_high;

  if (part_by_nnz_ || part_by_ncnfls_) {
    rowptr_low[0] = 0;
    rowptr_high[0] = 0;
    for (int i = 0; i < nrows_; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (abs(colind_[j] - i) < HybBwThreshold ||
            (row_part_[colind_[j]] == row_part_[i])) {
          rowptr_low[i + 1]++;
          colind_low.push_back(colind_[j]);
          values_low.push_back(values_[j]);
        } else {
          rowptr_high[i + 1]++;
          colind_high.push_back(colind_[j]);
          values_high.push_back(values_[j]);
        }
      }
    }
  } else {
    rowptr_low[0] = 0;
    rowptr_high[0] = 0;
    for (int i = 0; i < nrows_; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (abs(colind_[j] - i) < HybBwThreshold) {
          rowptr_low[i + 1]++;
          colind_low.push_back(colind_[j]);
          values_low.push_back(values_[j]);
        } else {
          rowptr_high[i + 1]++;
          colind_high.push_back(colind_[j]);
          values_high.push_back(values_[j]);
        }
      }
    }
  }

  for (int i = 1; i < nrows_ + 1; ++i) {
    rowptr_low[i] += rowptr_low[i - 1];
    rowptr_high[i] += rowptr_high[i - 1];
  }
  assert(rowptr_low[nrows_] == static_cast<int>(values_low.size()));
  assert(rowptr_high[nrows_] == static_cast<int>(values_high.size()));

  nnz_lbw_ = values_low.size();
  move(rowptr_low.begin(), rowptr_low.end(), rowptr_);
  move(colind_low.begin(), colind_low.end(), colind_);
  move(values_low.begin(), values_low.end(), values_);

  nnz_hbw_ = values_high.size();
  rowptr_h_ =
      (IndexT *)internal_alloc((nrows_ + 1) * sizeof(IndexT), platform_);
  colind_h_ = (IndexT *)internal_alloc(nnz_hbw_ * sizeof(IndexT), platform_);
  values_h_ = (ValueT *)internal_alloc(nnz_hbw_ * sizeof(ValueT), platform_);

  // Enforce first-touch memory affinity
  #pragma omp parallel num_threads(nthreads_)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i <= nrows_; i++) {
      rowptr_h_[i] = rowptr_high[i];
    }

    #pragma omp for schedule(static)
    for (int i = 0; i < nnz_hbw_; i++) {
      colind_h_[i] = colind_high[i];
      values_h_[i] = values_high[i];
    }
  }

#ifdef _LOG_INFO
  cout << fixed;
  cout << setprecision(2);
  cout << "[INFO]: " << ((float)nnz_hbw_ / nnz_) * 100
       << " % of nonzeros are in non-symmetric sub-matrix" << endl;
#endif

  split_by_bw_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::partition_by_nrows(int nthreads) {
#ifdef _LOG_INFO
  cout << "[INFO]: partitioning by number of rows" << endl;
#endif

  if (!row_split_) {
    row_split_ =
        (IndexT *)internal_alloc((nthreads + 1) * sizeof(IndexT), platform_);
  }

  // Re-init
  memset(row_split_, 0, (nthreads + 1) * sizeof(IndexT));

  // Compute new matrix splits
  int nrows_per_split = ((nrows_ / nthreads - 1) | (BlkFactor - 1)) + 1;
  row_split_[0] = 0;
  for (int i = 0; i < nthreads - 1; i++) {
    row_split_[i + 1] = row_split_[i] + nrows_per_split;
  }
  row_split_[nthreads] = nrows_;

  row_subset_.resize(nthreads_);
  row_part_ = (int *)internal_alloc(nrows_ * sizeof(int));
  for (int t = 0; t < nthreads_; ++t) {
    for (int i = row_split_[t]; i < row_split_[t + 1]; ++i) {
      row_subset_[t].push_back(i);
      row_part_[i] = t;
    }
  }

  part_by_nrows_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::partition_by_nnz(int nthreads) {
#ifdef _LOG_INFO
  cout << "[INFO]: partitioning by number of nonzeros" << endl;
#endif

  if (!row_split_) {
    row_split_ =
        (IndexT *)internal_alloc((nthreads + 1) * sizeof(IndexT), platform_);
  }

  if (nthreads_ == 1) {
    row_split_[0] = 0;
    row_split_[1] = nrows_;
    part_by_nnz_ = true;
    return;
  }

  // Compute the matrix splits.
  int nnz_cnt = (symmetric_) ? (nnz_ - nrows_) / 2 : nnz_;
  int nnz_per_split = nnz_cnt / nthreads_;
  int curr_nnz = 0;
  int row_start = 0;
  int split_cnt = 0;
  int i;

  row_split_[0] = row_start;
  if (hybrid_ && split_by_bw_) {
    nnz_cnt = (nnz_lbw_ - nrows_) / 2 + nnz_hbw_;
    nnz_per_split = nnz_cnt / nthreads_;
    for (i = 0; i < nrows_; i++) {
      int row_nnz = 0;
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          row_nnz++;
        }
      }
      row_nnz += rowptr_h_[i + 1] - rowptr_h_[i];
      curr_nnz += row_nnz;

      if ((curr_nnz >= nnz_per_split)) { // && ((i + 1) % BlkFactor == 0)) {
        row_start = i + 1;
        ++split_cnt;
        if (split_cnt <= nthreads)
          row_split_[split_cnt] = row_start;
        curr_nnz = 0;
      }
    }
  } else if (symmetric_) {
    for (i = 0; i < nrows_; i++) {
      int row_nnz = 0;
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          row_nnz++;
        }
      }
      curr_nnz += row_nnz;

      if ((curr_nnz >= nnz_per_split) && ((i + 1) % BlkFactor == 0)) {
        row_start = i + 1;
        ++split_cnt;
        if (split_cnt <= nthreads)
          row_split_[split_cnt] = row_start;
        curr_nnz = 0;
      }
    }
  } else {
    for (i = 0; i < nrows_; i++) {
      curr_nnz += rowptr_[i + 1] - rowptr_[i];
      if ((curr_nnz >= nnz_per_split) && ((i + 1) % BlkFactor == 0)) {
        row_start = i + 1;
        ++split_cnt;
        if (split_cnt <= nthreads)
          row_split_[split_cnt] = row_start;
        curr_nnz = 0;
      }
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

  row_subset_.resize(nthreads_);
  row_part_ = (int *)internal_alloc(nrows_ * sizeof(int));
  for (int t = 0; t < nthreads_; ++t) {
    for (int i = row_split_[t]; i < row_split_[t + 1]; ++i) {
      row_subset_[t].push_back(i);
      row_part_[i] = t;
    }
  }

  part_by_nnz_ = true;
}

#if defined(_METIS) || defined(_KAHIP)
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::partition_by_conflicts(int nthreads) {
  assert(symmetric_);
#ifdef _LOG_INFO
  cout << "[INFO]: partitioning by number of nonzeros so that direct conflicts "
          "are minimized"
       << endl;
  float tstart = omp_get_wtime();
#endif

  IndexT *xadj;   // vertices
  IndexT *adjncy; // edges
  IndexT *vwgt;   // vertex weights
  std::vector<int> v_adjncy;
  int V = (int)ceil(nrows_ / (double)BlkFactor);
  vwgt = (int *)internal_alloc(V * sizeof(int));
  if (BlkFactor == 1) {
    xadj = rowptr_;
    adjncy = colind_;
    for (int i = 0; i < V; ++i) {
      vwgt[i] = rowptr_[i + 1] - rowptr_[i];
    }
  } else {
    std::set<int> tmp;
    xadj = (IndexT *)internal_alloc((V + 1) * sizeof(IndexT));
    xadj[0] = 0;
    for (int i = 0; i < nrows_; i += BlkFactor) {
      for (int k = i; k < min(i + BlkFactor, nrows_); ++k) {
        for (int j = rowptr_[k]; j < rowptr_[k + 1]; ++j) {
          tmp.insert(colind_[j] >> BlkBits);
        }
      }

      for (auto &elem : tmp)
        v_adjncy.push_back(elem);
      xadj[(i >> BlkBits) + 1] = xadj[(i >> BlkBits)] + tmp.size();
      vwgt[i >> BlkBits] = rowptr_[min(i + BlkFactor, nrows_)] - rowptr_[i];
      tmp.clear();
    }

    adjncy = v_adjncy.data();
  }

#ifdef _METIS
  // Do k-way balanced partitioning with METIS
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_NITER] = 1;
  options[METIS_OPTION_MINCONN] = 1;
  options[METIS_OPTION_UFACTOR] = 10;
  options[METIS_OPTION_DBGLVL] = 0;
  int ncon = 1; // balancing constraints
  idx_t edgecut;
  row_part_ = (int *)internal_alloc(V * sizeof(int));
  int ret =
      METIS_PartGraphKway(&V, &ncon, xadj, adjncy, vwgt, NULL, NULL, &nthreads_,
                          NULL, NULL, options, &edgecut, row_part_);
  if (ret != METIS_OK) {
    cout << "[ERROR]: METIS failed" << endl;
    exit(1);
  }
#endif // _METIS

#ifdef _KAHIP
  // Do k-way balanced partitioning with KaHIP
  double imbalance = 0.01;
  int edgecut;
  bool suppress_output = false;
  row_part_ = (int *)internal_alloc(V * sizeof(int));
  kaffpa(&V, vwgt, xadj, NULL, adjncy, &nthreads_, &imbalance, suppress_output,
         0, STRONG, &edgecut, row_part_);
#endif // _KAHIP

  // Convert information to (thread, row subset) mapping
  row_subset_.resize(nthreads_);
  for (int i = 0; i < nrows_; ++i) {
    row_subset_[row_part_[i >> BlkBits]].push_back(i);
  }

  // Cleanup
  internal_free(vwgt);
  if (BlkFactor > 1) {
    internal_free(xadj);
    adjncy = nullptr;
    v_adjncy.clear();
  }

#ifdef _LOG_INFO
  float tstop = omp_get_wtime();
  cout << "[INFO]: graph partitioning: " << tstop - tstart << endl;
#endif

  part_by_ncnfls_ = true;
}
#endif

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::serial() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry" << endl;
#endif

  sym_thread_data_ = (SymThreadData **)internal_alloc(sizeof(SymThreadData *));
  sym_thread_data_[0] = new SymThreadData;
  SymThreadData *data = sym_thread_data_[0];
  data->nrows_ = nrows_;
  data->rowptr_ = (IndexT *)internal_alloc((nrows_ + 1) * sizeof(IndexT));
  memset(data->rowptr_, 0, (nrows_ + 1) * sizeof(IndexT));
  data->diagonal_ = (ValueT *)internal_alloc(nrows_ * sizeof(ValueT));
  memset(data->diagonal_, 0, nrows_ * sizeof(IndexT));

  vector<IndexT> colind_sym;
  vector<ValueT> values_sym;
  int nnz_diag = 0;
  int nnz_estimated = nnz_ / 2;
  colind_sym.reserve(nnz_estimated);
  values_sym.reserve(nnz_estimated);
  data->rowptr_[0] = 0;
  for (int i = 0; i < nrows_; ++i) {
    for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
      if (colind_[j] < i) {
        data->rowptr_[i + 1]++;
        colind_sym.push_back(colind_[j]);
        values_sym.push_back(values_[j]);
      } else if (colind_[j] == i) {
        data->diagonal_[i] = values_[j];
        nnz_diag++;
      }
    }
  }

  for (int i = 1; i <= nrows_; ++i) {
    data->rowptr_[i] += data->rowptr_[i - 1];
  }

  assert(data->rowptr_[nrows_] == static_cast<int>(values_sym.size()));
  nnz_low_ = data->nnz_low_ = values_sym.size();
  nnz_diag_ = data->nnz_diag_ = nnz_diag;
  data->nnz_ = data->nnz_low_ + data->nnz_diag_;
  data->colind_ =
      (IndexT *)internal_alloc(data->nnz_low_ * sizeof(IndexT), platform_);
  data->values_ =
      (ValueT *)internal_alloc(data->nnz_low_ * sizeof(ValueT), platform_);

  for (int i = 0; i < nrows_; ++i) {
    for (int j = data->rowptr_[i]; j < data->rowptr_[i + 1]; ++j) {
      data->colind_[j] = colind_sym[j];
      data->values_[j] = values_sym[j];
    }
  }

  // Cleanup
  colind_sym.clear();
  values_sym.clear();
  colind_sym.shrink_to_fit();
  values_sym.shrink_to_fit();

  cmp_symmetry_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::atomics() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using atomics" << endl;
#endif

  sym_thread_data_ =
      (SymThreadData **)internal_alloc(nthreads_ * sizeof(SymThreadData *));
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    sym_thread_data_[tid] = new SymThreadData;
    SymThreadData *data = sym_thread_data_[tid];
    int nrows = row_split_[tid + 1] - row_split_[tid];
    int row_offset = row_split_[tid];
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    int nnz_diag = 0;
    int nnz_estimated =
        (rowptr_[row_split_[tid + 1]] - rowptr_[row_split_[tid]]) / 2;
    colind_sym.reserve(nnz_estimated);
    values_sym.reserve(nnz_estimated);

    data->rowptr_[0] = 0;
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          data->rowptr_[i + 1 - row_offset]++;
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          data->diagonal_[i - row_offset] = values_[j];
          nnz_diag++;
        }
      }
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_low_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->nnz_ = data->nnz_low_ + data->nnz_diag_;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_low_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_low_ * sizeof(ValueT), platform_);

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
    colind_sym.shrink_to_fit();
    values_sym.shrink_to_fit();
  }

  for (int t = 0; t < nthreads_; ++t) {
    SymThreadData *data = sym_thread_data_[t];
    nnz_low_ += data->nnz_low_;
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

  sym_thread_data_ =
      (SymThreadData **)internal_alloc(nthreads_ * sizeof(SymThreadData *));
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    sym_thread_data_[tid] = new SymThreadData;
    SymThreadData *data = sym_thread_data_[tid];
    int nrows = row_split_[tid + 1] - row_split_[tid];
    int row_offset = row_split_[tid];
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    int nnz_diag = 0;
    int nnz_estimated =
        (rowptr_[row_split_[tid + 1]] - rowptr_[row_split_[tid]]) / 2;
    colind_sym.reserve(nnz_estimated);
    values_sym.reserve(nnz_estimated);

    data->rowptr_[0] = 0;
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          data->rowptr_[i + 1 - row_offset]++;
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          data->diagonal_[i - row_offset] = values_[j];
          nnz_diag++;
        }
      }
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_low_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->nnz_ = data->nnz_low_ + data->nnz_diag_;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_low_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_low_ * sizeof(ValueT), platform_);

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
    colind_sym.shrink_to_fit();
    values_sym.shrink_to_fit();
  }

  for (int t = 0; t < nthreads_; ++t) {
    SymThreadData *data = sym_thread_data_[t];
    nnz_low_ += data->nnz_low_;
    nnz_diag_ += data->nnz_diag_;
  }

  cmp_symmetry_ = true;
  effective_ranges_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::local_vectors_indexing() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using local vectors indexing"
       << endl;
#endif

  y_local_ = (ValueT **)internal_alloc(nthreads_ * sizeof(ValueT *), platform_);
  sym_thread_data_ =
      (SymThreadData **)internal_alloc(nthreads_ * sizeof(SymThreadData *));
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    sym_thread_data_[tid] = new SymThreadData;
    SymThreadData *data = sym_thread_data_[tid];
    int nrows = row_split_[tid + 1] - row_split_[tid];
    int row_offset = row_split_[tid];
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    int nnz_diag = 0;
    int nnz_estimated =
        (rowptr_[row_split_[tid + 1]] - rowptr_[row_split_[tid]]) / 2;
    colind_sym.reserve(nnz_estimated);
    values_sym.reserve(nnz_estimated);

    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          data->rowptr_[i + 1 - row_offset]++;
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          data->diagonal_[i - row_offset] = values_[j];
          nnz_diag++;
        }
      }
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_low_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->nnz_ = data->nnz_low_ + data->nnz_diag_;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_low_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_low_ * sizeof(ValueT), platform_);

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
    colind_sym.shrink_to_fit();
    values_sym.shrink_to_fit();
  }

  for (int t = 0; t < nthreads_; ++t) {
    SymThreadData *data = sym_thread_data_[t];
    nnz_low_ += data->nnz_low_;
    nnz_diag_ += data->nnz_diag_;
  }

  cmp_symmetry_ = true;

  if (nthreads_ == 1)
    return;

  // Global map of conflicts
  map<IndexT, set<int>> global_map;
  // Conflicting rows per thread
  set<IndexT> thread_map;
  int ncnfls = 0;
  for (int tid = 1; tid < nthreads_; tid++) {
    SymThreadData *data = sym_thread_data_[tid];
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
#ifdef _LOG_INFO
  cout << "[INFO]: detected " << ncnfls << " conflicts" << endl;
#endif

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
      SymThreadData *data = sym_thread_data_[tid];
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
    SymThreadData *data = sym_thread_data_[tid];
    data->map_start_ = start;
    if (tid < nthreads_ - 1)
      data->map_end_ += start;
    start = data->map_end_;
    if (tid == nthreads_ - 1)
      assert(data->map_end_ = ncnfls);
  }

  local_vectors_indexing_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::conflict_free_apriori() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using a priori conflict-free SpMV"
       << endl;
#endif

  // FIXME: first touch
  sym_thread_data_ = (SymThreadData **)internal_alloc(sizeof(SymThreadData *));
  sym_thread_data_[0] = new SymThreadData;
  SymThreadData *data = sym_thread_data_[0];
  data->nrows_ = nrows_;
  data->rowptr_ = (IndexT *)internal_alloc((nrows_ + 1) * sizeof(IndexT));
  memset(data->rowptr_, 0, (nrows_ + 1) * sizeof(IndexT));
  data->diagonal_ = (ValueT *)internal_alloc(nrows_ * sizeof(ValueT));
  memset(data->diagonal_, 0, nrows_ * sizeof(IndexT));

  vector<IndexT> colind_sym;
  vector<ValueT> values_sym;
  int nnz_estimated = nnz_ / 2;
  colind_sym.reserve(nnz_estimated);
  values_sym.reserve(nnz_estimated);

  nnz_diag_ = 0;
  data->rowptr_[0] = 0;
  for (int tid = 0; tid < nthreads_; ++tid) {
    for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
        if (colind_[j] < i) {
          data->rowptr_[i + 1]++;
          colind_sym.push_back(colind_[j]);
          values_sym.push_back(values_[j]);
        } else if (colind_[j] == i) {
          data->diagonal_[i] = values_[j];
          nnz_diag_++;
        }
      }
    }
  }

  for (int i = 1; i <= nrows_; ++i) {
    data->rowptr_[i] += data->rowptr_[i - 1];
  }

  assert(data->rowptr_[nrows_] == static_cast<int>(values_sym.size()));
  nnz_low_ = data->nnz_low_ = values_sym.size();
  data->colind_ =
      (IndexT *)internal_alloc(nnz_low_ * sizeof(IndexT), platform_);
  data->values_ =
      (ValueT *)internal_alloc(nnz_low_ * sizeof(ValueT), platform_);

  #pragma omp parallel for schedule(static) num_threads(nthreads_)
  for (int j = 0; j < nnz_low_; ++j) {
    data->colind_[j] = colind_sym[j];
    data->values_[j] = values_sym[j];
  }

  // Cleanup
  colind_sym.clear();
  values_sym.clear();
  colind_sym.shrink_to_fit();
  values_sym.shrink_to_fit();

  cmp_symmetry_ = true;

  if (nthreads_ == 1)
    return;

#ifdef _LOG_INFO
  float assembly_start = omp_get_wtime();
#endif
  const int blk_rows = (int)ceil(nrows_ / (double)BlkFactor);
  ConflictGraph g(blk_rows);
  vector<WeightedVertex> vertices(blk_rows);
  tbb::concurrent_vector<tbb::concurrent_vector<int>> indirect(blk_rows);

  // Add direct conflicts
  for (int i = 0; i < nrows_; ++i) {
    IndexT blk_row = i >> BlkBits;
    IndexT prev_blk_col = -1;
    for (int j = data->rowptr_[i]; j < data->rowptr_[i + 1]; ++j) {
      IndexT blk_col = data->colind_[j] >> BlkBits;
      // g[blk_row].insert(blk_col);
      // g[blk_col].insert(blk_row);
      // Mark potential indirect conflicts
      if (blk_col != prev_blk_col)
        indirect[blk_col].push_back(blk_row);
      prev_blk_col = blk_col;
    }
  }

  // Add indirect conflicts
  // Indirect conflicts occur when two rows have nonzero elements in the same
  // column.
  for (int i = 0; i < blk_rows; i++) {
    for (const auto &row1 : indirect[i]) {
      for (const auto &row2 : indirect[i]) {
        if (row1 < row2) {
          // g[row1].insert(row2);
          // g[row2].insert(row1);
        }
      }
    }
  }

  for (auto &i : indirect)
    i.clear();
  indirect.clear();

#ifdef _LOG_INFO
  float assembly_stop = omp_get_wtime();
  cout << "[INFO]: graph assembly: " << assembly_stop - assembly_start << endl;
  cout << "[INFO]: using a blocking factor of: " << BlkFactor << endl;
#endif

  const int V = g.size();
  vector<int> color_map(V, V - 1);
  color_greedy(g, {}, false, color_map);

  // Find row indices per color
  vector<vector<IndexT>> rowind(ncolors_);
  for (int i = 0; i < nrows_; i++) {
    rowind[color_map[i >> BlkBits]].push_back(i);
  }

  // Allocate auxiliary arrays
  data->color_ptr_ = (IndexT *)internal_alloc((ncolors_ + 1) * sizeof(IndexT));
  memset(data->color_ptr_, 0, (ncolors_ + 1) * sizeof(IndexT));
  for (int c = 1; c <= ncolors_; ++c) {
    data->color_ptr_[c] += data->color_ptr_[c - 1] + rowind[c - 1].size();
  }
  assert(data->color_ptr_[ncolors_] == nrows_);

  data->rowind_ = (IndexT *)internal_alloc(nrows_ * sizeof(IndexT));
  int cnt = 0;
  for (int c = 0; c < ncolors_; ++c) {
    sort(rowind[c].begin(), rowind[c].end());
    for (size_t i = 0; i < rowind[c].size(); ++i) {
      data->rowind_[cnt++] = rowind[c][i];
    }
  }

  conflict_free_apriori_ = true;
}

// Assumes rows have been assigned contiguously to threads
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::conflict_free_aposteriori() {
  // Sanity check
  assert(symmetric_);

#ifdef _LOG_INFO
  cout << "[INFO]: compressing for symmetry using a posteriori conflict-free "
          "SpMV"
       << endl;
#endif

  sym_thread_data_ =
      (SymThreadData **)internal_alloc(nthreads_ * sizeof(SymThreadData *));
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    sym_thread_data_[tid] = new SymThreadData;
    SymThreadData *data = sym_thread_data_[tid];
    int nrows = 0;
    if (part_by_ncnfls_) {
      nrows = row_subset_[tid].size();
      data->rowind_ = (IndexT *)internal_alloc(nrows * sizeof(IndexT));
    } else {
      nrows = row_split_[tid + 1] - row_split_[tid];
      data->row_offset_ = row_split_[tid];
    }
    data->nrows_ = nrows;
    data->rowptr_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
    memset(data->rowptr_, 0, (nrows + 1) * sizeof(IndexT));
    data->diagonal_ = (ValueT *)internal_alloc(nrows * sizeof(ValueT));
    memset(data->diagonal_, 0, nrows * sizeof(IndexT));

    vector<IndexT> colind_sym;
    vector<ValueT> values_sym;
    int nnz_diag = 0;
    data->rowptr_[0] = 0;
    if (part_by_ncnfls_) {
      for (int i = 0; i < nrows; ++i) {
        int row = row_subset_[tid][i];
        data->rowind_[i] = row;
        for (int j = rowptr_[row]; j < rowptr_[row + 1]; ++j) {
          IndexT col = colind_[j];
          if (col < row) {
            data->rowptr_[i + 1]++;
            colind_sym.push_back(col);
            values_sym.push_back(values_[j]);
          } else if (col == row) {
            data->diagonal_[i] = values_[j];
            nnz_diag++;
          }
        }
      }
    } else {
      int nnz_estimated =
          (rowptr_[row_split_[tid + 1]] - rowptr_[row_split_[tid]]) / 2;
      colind_sym.reserve(nnz_estimated);
      values_sym.reserve(nnz_estimated);
      int row_offset = row_split_[tid];
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
    }

    for (int i = 1; i <= nrows; ++i) {
      data->rowptr_[i] += data->rowptr_[i - 1];
    }

    assert(data->rowptr_[nrows] == static_cast<int>(values_sym.size()));
    data->nnz_low_ = values_sym.size();
    data->nnz_diag_ = nnz_diag;
    data->nnz_ = data->nnz_low_ + data->nnz_diag_;
    data->colind_ =
        (IndexT *)internal_alloc(data->nnz_low_ * sizeof(IndexT), platform_);
    data->values_ =
        (ValueT *)internal_alloc(data->nnz_low_ * sizeof(ValueT), platform_);

    move(colind_sym.begin(), colind_sym.end(), data->colind_);
    move(values_sym.begin(), values_sym.end(), data->values_);

    // Cleanup
    colind_sym.clear();
    values_sym.clear();
    colind_sym.shrink_to_fit();
    values_sym.shrink_to_fit();

    if (hybrid_) {
      data->rowptr_h_ = (IndexT *)internal_alloc((nrows + 1) * sizeof(IndexT));
      memset(data->rowptr_h_, 0, (nrows + 1) * sizeof(IndexT));
      vector<IndexT> colind_high;
      vector<ValueT> values_high;
      data->rowptr_h_[0] = 0;
      if (part_by_ncnfls_) {
        for (int i = 0; i < nrows; ++i) {
          int row = row_subset_[tid][i];
          for (int j = rowptr_h_[row]; j < rowptr_h_[row + 1]; ++j) {
            data->rowptr_h_[i + 1]++;
            colind_high.push_back(colind_h_[j]);
            values_high.push_back(values_h_[j]);
          }
        }
      } else {
        int row_offset = row_split_[tid];
        for (int i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
          for (int j = rowptr_h_[i]; j < rowptr_h_[i + 1]; ++j) {
            data->rowptr_h_[i + 1 - row_offset]++;
            colind_high.push_back(colind_h_[j]);
            values_high.push_back(values_h_[j]);
          }
        }
      }

      for (int i = 1; i <= nrows; ++i) {
        data->rowptr_h_[i] += data->rowptr_h_[i - 1];
      }
      data->nnz_hbw_ = values_high.size();
      data->colind_h_ =
          (IndexT *)internal_alloc(data->nnz_hbw_ * sizeof(IndexT), platform_);
      data->values_h_ =
          (ValueT *)internal_alloc(data->nnz_hbw_ * sizeof(ValueT), platform_);
      move(colind_high.begin(), colind_high.end(), data->colind_h_);
      move(values_high.begin(), values_high.end(), data->values_h_);

      // Cleanup
      colind_high.clear();
      values_high.clear();
      colind_high.shrink_to_fit();
      values_high.shrink_to_fit();
    }

    if (part_by_ncnfls_) {
      row_subset_[tid].clear();
      row_subset_[tid].shrink_to_fit();
    }
  }

  for (int t = 0; t < nthreads_; ++t) {
    SymThreadData *data = sym_thread_data_[t];
    nnz_low_ += data->nnz_low_;
    nnz_diag_ += data->nnz_diag_;
  }

  cmp_symmetry_ = true;

  if (nthreads_ == 1)
    return;

#ifdef _LOG_INFO
  float assembly_start = omp_get_wtime();
#endif
  const int blk_rows = (int)ceil(nrows_ / (double)BlkFactor);
  ConflictGraph g(blk_rows);
  vector<WeightedVertex> vertices(blk_rows);

  if (part_by_ncnfls_) {
    tbb::concurrent_vector<tbb::concurrent_vector<pair<int, int>>> indirect(
        blk_rows);
    #pragma omp parallel num_threads(nthreads_)
    {
      int tid = omp_get_thread_num();
      SymThreadData *data = sym_thread_data_[tid];
      int nrows = data->nrows_;
      IndexT row_offset = 0;
      for (int i = 0; i < nrows; ++i) {
        int row = data->rowind_[i];
        IndexT blk_row = row >> BlkBits;
        vertices[blk_row].vid = blk_row;
        vertices[blk_row].tid = tid;
        vertices[blk_row].weight +=
            data->rowptr_[i - row_offset + 1] - data->rowptr_[i - row_offset];
        if (hybrid_)
          vertices[blk_row].weight += data->rowptr_h_[i - row_offset + 1] -
                                      data->rowptr_h_[i - row_offset];
        IndexT prev_blk_col = -1;
        for (int j = data->rowptr_[i - row_offset];
             j < data->rowptr_[i + 1 - row_offset]; ++j) {
          IndexT col = data->colind_[j];
          IndexT blk_col = col >> BlkBits;
          // If this nonzero is in the lower triangular part and has a direct
          // conflict with another thread
          if (row_part_[blk_col] != row_part_[blk_row]) {
            g[blk_row].insert(blk_col);
            g[blk_col].insert(blk_row);
          }

          // Mark potential indirect conflicts
          if (blk_col != prev_blk_col)
            indirect[blk_col].push_back(make_pair(blk_row, tid));
          prev_blk_col = blk_col;
        }
      }

      #pragma omp barrier

      for (int j = 0; j < nrows; j++) {
        int i = data->rowind_[j] >> BlkBits;
        for (const auto &row1 : indirect[i]) {
          for (const auto &row2 : indirect[i]) {
            if ((row1.first < row2.first) && (row1.second != row2.second)) {
              g[row1.first].insert(row2.first);
              g[row2.first].insert(row1.first);
            }
          }
        }
      }
    }

    // Cleanup
    for (auto &i : indirect)
      i.clear();
    indirect.clear();
    indirect.shrink_to_fit();
  } else {
    #pragma omp parallel num_threads(nthreads_)
    {
      int tid = omp_get_thread_num();
      SymThreadData *data = sym_thread_data_[tid];
      IndexT row_offset = row_split_[tid];
      for (int i = row_split_[tid]; i < row_split_[tid + 1]; i++) {
        IndexT blk_row = i >> BlkBits;
        vertices[blk_row].vid = blk_row;
        vertices[blk_row].tid = tid;
        vertices[blk_row].weight +=
            data->rowptr_[i - row_offset + 1] - data->rowptr_[i - row_offset];
        if (hybrid_)
          vertices[blk_row].weight += data->rowptr_h_[i - row_offset + 1] -
                                      data->rowptr_h_[i - row_offset];

        // Direct conflicts
        for (int j = data->rowptr_[i - row_offset];
             j < data->rowptr_[i - row_offset + 1]; ++j) {
          IndexT col = data->colind_[j];
          IndexT blk_col = col >> BlkBits;
          if (col < row_offset) {
            g[blk_row].insert(blk_col);
            g[blk_col].insert(blk_row);
          }
        }

        // Mark first nonzero in upper triangular matrix for this row
        int first_upper = rowptr_[i + 1];
        for (int j = rowptr_[i]; j < rowptr_[i + 1]; ++j) {
          if (colind_[j] > i) {
            first_upper = j;
            break;
          }
        }

        // Indirect conflicts, treat rows as columns
        for (int j = first_upper + 1; j < rowptr_[i + 1]; ++j) {
          IndexT cur_row = colind_[j];
          for (int k = first_upper; k < j; ++k) {
            IndexT prev_row = colind_[k];
            // If they belong to different threads
            if (row_part_[prev_row] != row_part_[cur_row]) {
              int prev_blk_row = prev_row >> BlkBits;
              int cur_blk_row = cur_row >> BlkBits;
              g[prev_blk_row].insert(cur_blk_row);
              g[cur_blk_row].insert(prev_blk_row);
            }
          }
        }
      }
    }
  }

#ifdef _LOG_INFO
  float assembly_stop = omp_get_wtime();
  cout << "[INFO]: graph assembly: " << assembly_stop - assembly_start << endl;
  cout << "[INFO]: using a blocking factor of: " << BlkFactor << endl;
#endif

#ifdef _REPORT_DETAILS
  // Compute maximum vertex degree
  int max_vertex_degree = 0;
  for (const auto &v : g) {
    if (static_cast<int>(v.size()) > max_vertex_degree)
      max_vertex_degree = v.size();
  }
  cout << "[INFO]: maximum vertex degree: " << max_vertex_degree << endl;
#endif

  const int V = g.size();
  vector<int> color_map(V, V - 1);
  if (part_by_ncnfls_)
    color_greedy(g, vertices, false, color_map);
  else
    color_greedy(g, vertices, true, color_map);

#ifndef _USE_BARRIER
  // Find thread dependency graph between colors.
  // Need to check if a row in color C is touched in color C-1 by others threads
  // It is enough to check if the neighbors of the corresponding vertex are
  // colored with the previous color and are assigned to different threads
  bool cnfls[MaxColors][MaxThreads][MaxThreads];
  for (int i = 0; i < ncolors_; i++) {
    for (int t1 = 0; t1 < nthreads_; t1++) {
      for (int t2 = 0; t2 < nthreads_; t2++) {
        cnfls[i][t1][t2] = false;
      }
    }
  }

  // FIXME: optimize
  for (int i = 0; i < V; i++) {
    int c_i = color_map[i];
    if (c_i > 0) {
      const auto &neighbors = g[i];
      for (auto j : neighbors) {
        int c_j = color_map[j];
        // Mark who I need to wait for before proceeding to current color
        if ((c_j == (c_i - 1)) && (vertices[i].tid != vertices[j].tid))
          cnfls[c_i][vertices[i].tid][vertices[j].tid] = true;
      }
    }
  }

#ifdef _LOG_INFO
  cout << "[INFO]: Dependency graph" << endl;
  for (int i = 0; i < ncolors_; i++) {
    for (int t1 = 0; t1 < nthreads_; t1++) {
      for (int t2 = 0; t2 < nthreads_; t2++) {
        if (cnfls[i][t1][t2])
          cout << "\t(C" << i << ", T" << t1 << ", T" << t2 << ")" << endl;
      }
    }
  }
#endif
#endif

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    SymThreadData *data = sym_thread_data_[tid];

#ifndef _USE_BARRIER
    // Determine dependency graph
    for (int c = 0; c < ncolors_; c++) {
      for (int t = 0; t < nthreads_; t++) {
        if (cnfls[c][tid][t]) {
          data->deps_[c].push_back(t);
        }
      }
    }
#endif

    // Find active row indices per color
    vector<vector<IndexT>> rowind(ncolors_);
    if (part_by_ncnfls_) {
      int nrows = data->nrows_;
      for (int i = 0; i < nrows; ++i) {
        rowind[color_map[data->rowind_[i] >> BlkBits]].push_back(i);
      }
    } else {
      for (int i = row_split_[tid]; i < row_split_[tid + 1]; i++) {
        rowind[color_map[i >> BlkBits]].push_back(i);
      }
    }

    // Detect ranges of consecutive rows
    vector<vector<IndexT>> row_start(ncolors_ + 1);
    vector<vector<IndexT>> row_end(ncolors_ + 1);
    IndexT row, row_prev;
    int nranges = 0;
    for (int c = 0; c < ncolors_; c++) {
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
    rowind.shrink_to_fit();

    // Allocate auxiliary arrays
    data->nranges_ = nranges;
    data->range_ptr_ =
        (IndexT *)internal_alloc((ncolors_ + 1) * sizeof(IndexT));
    data->range_start_ = (IndexT *)internal_alloc(nranges * sizeof(IndexT));
    data->range_end_ = (IndexT *)internal_alloc(nranges * sizeof(IndexT));

    int cnt = 0;
#if defined(_METIS) || defined(_KAHIP)
    int row_offset = 0;
#else
    int row_offset = row_split_[tid];
#endif
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

  for (int t = 0; t < nthreads_; ++t) {
    SymThreadData *data = sym_thread_data_[t];
    nranges_ += data->nranges_;
  }

#if defined(_REPORT_DETAILS)
  estimate_imbalance();
#endif

  conflict_free_aposteriori_ = true;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::estimate_imbalance() {
  vector<int> thread_nnz(nthreads_);
  vector<float> m_c(ncolors_);
  vector<float> I_c(ncolors_);
  float I = 0;

  for (int c = 0; c < ncolors_; c++) {
    int total_nnz_c = 0;
    for (int t = 0; t < nthreads_; t++) {
      SymThreadData *data = sym_thread_data_[t];
      thread_nnz[t] = 0;
      for (int r = data->range_ptr_[c]; r < data->range_ptr_[c + 1]; ++r) {
        for (IndexT i = data->range_start_[r]; i <= data->range_end_[r]; ++i) {
          thread_nnz[t] += data->rowptr_[i + 1] - data->rowptr_[i];
        }
      }
      total_nnz_c += thread_nnz[t];
    }

    m_c[c] = total_nnz_c / nthreads_;
    float max_I_c_p = 0;
    for (int t = 0; t < nthreads_; t++) {
      // Absolute nonzero imbalance
      float I_c_p = abs(thread_nnz[t] - m_c[c]);
      if (I_c_p > max_I_c_p)
        max_I_c_p = I_c_p;
    }
    I_c[c] = max_I_c_p * 100;
  }

  float I_c_1 = 0;
  float I_c_2 = 0;
  for (int c = 0; c < ncolors_; c++) {
    I_c_1 += I_c[c];
    I_c_2 += m_c[c];
  }

  I = I_c_1 / I_c_2;
  cout << "[INFO]: I: " << I << " %" << endl;
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::compress_symmetry() {
  if (!symmetric_) {
    return;
  }

  if (nthreads_ == 1) {
    serial();
  } else {
    // atomics();
    // effective_ranges();
    // local_vectors_indexing();
    // conflict_free_apriori();
    conflict_free_aposteriori();
  }

  // Cleanup
  if (owns_data_) {
    internal_free(rowptr_, platform_);
    internal_free(colind_, platform_);
    internal_free(values_, platform_);
    rowptr_ = nullptr;
    colind_ = nullptr;
    values_ = nullptr;
    if (hybrid_) {
      internal_free(rowptr_h_, platform_);
      internal_free(colind_h_, platform_);
      internal_free(values_h_, platform_);
      rowptr_h_ = nullptr;
      colind_h_ = nullptr;
      values_h_ = nullptr;
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::first_fit_round_robin(const ConflictGraph &g,
                                                      vector<int> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying FF-RR vertex ordering..." << endl;
#endif

  int cnt = 0, t_cnt = 0;
  while ((unsigned int)cnt < g.size()) {
    for (int t = 0; t < nthreads_; t++) {
      if (row_split_[t] + t_cnt < row_split_[t + 1]) {
        assert(((row_split_[t] + t_cnt) / BlkFactor) < nrows_);
        order.push_back((row_split_[t] + t_cnt) / BlkFactor);
        cnt++;
      }
    }

    t_cnt += BlkFactor;
  }

  assert(order.size() == g.size());
}

// Sort vertics by decreasing degree
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::largest_first(const ConflictGraph &g,
                                              vector<int> &order) {
#ifdef _LOG_INFO
  cout << "[INFO]: applying LF vertex ordering..." << endl;
#endif

  vector<WeightedVertex> degree(g.size());
  for (size_t i = 0; i < g.size(); ++i) {
    degree[i] = WeightedVertex(i, -1, g[i].size());
  }
  sort(degree.begin(), degree.end(), DecreasingVertexWeight());

  for (size_t i = 0; i < degree.size(); ++i) {
    order.emplace_back(degree[i].vid);
  }

  degree.clear();
  // degree.shrink_to_fit();
  assert(order.size() == g.size());
}

// LF:           Colors vertices in decreasing degree order.
// FF-RR:        Colors vertices in a round-robin fashion among threads
//               but in the order they appear in the graph representation.
template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::ordering_heuristic(const ConflictGraph &g,
                                                   vector<int> &order) {
  order.reserve(g.size());
  largest_first(g, order);
  // first_fit_round_robin(g, order);
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::color_dsatur(const ConflictGraph &g,
                                             const vector<WeightedVertex> &v,
                                             bool balance, vector<int> &color) {
  assert(symmetric_ && cmp_symmetry_);
#ifdef _LOG_INFO
  cout << "[INFO]: applying distance-1 graph coloring to detect "
          "conflict-free submatrices"
       << endl;
  float color_start = omp_get_wtime();
#endif

  const int V = g.size();
  unordered_map<int, unordered_set<int>> adj_colors;
  unordered_map<int, unordered_set<int>> adj_uncolored;
  vector<map<int, unordered_set<int>, greater<int>>> Q(MaxThreads);
  for (int i = 0; i < V; i++) {
    for (auto &neighbor : g[i]) {
      adj_uncolored[i].insert(neighbor);
    }
    Q[0][adj_uncolored[i].size()].insert(i);
  }

  int s = 0;
  int max_color = 0;
  while (s >= 0) {
    // Select vertex with maximum saturation degree and maximum effective degree
    auto it = Q[s].begin();
    int effective_degree = it->first;
    unordered_set<int> &elems = it->second;
    int v = *elems.begin();
    elems.erase(elems.begin());
    // If there are no more vertices left with this effective degree, remove
    // entry
    if (elems.empty())
      Q[s].erase(effective_degree);
    // Find the lowest color that is not being used by any of the most saturated
    // nodes neighbors, then color the most saturated node
    int lowest_color = 0;
    int done = 0;
    while (!done) {
      done = 1;
      for (auto &c : adj_colors[v]) {
        if (c == lowest_color) {
          lowest_color += 1;
          done = 0;
        }
      }
    }
    color[v] = lowest_color;
    if (lowest_color > max_color)
      max_color = lowest_color;

    for (auto &u : adj_uncolored[v]) {
      Q[adj_colors[u].size()][adj_uncolored[u].size()].erase(u);
      // If there are no more vertices left with this effective degree, remove
      // entry
      if (Q[adj_colors[u].size()][adj_uncolored[u].size()].empty())
        Q[adj_colors[u].size()].erase(adj_uncolored[u].size());
      adj_colors[u].insert(lowest_color);
      adj_uncolored[u].erase(v);
      s = max(s, static_cast<int>(adj_colors[u].size()));
      Q[adj_colors[u].size()][adj_uncolored[u].size()].insert(u);
    }
    while (s >= 0 && Q[s].empty()) {
      s--;
    }
  }

  ncolors_ = max_color + 1;
  for (int i = 0; i < V; i++) {
    assert(color[i] < ncolors_);
  }

  // Balancing phase
  if (balance) {
    // Find color class sizes in number of nonzeros (histogram
    // computation)
    vector<int> load_nnz(ncolors_, 0);
    if (!v.empty()) {
      for (int i = 0; i < V; ++i) {
        load_nnz[color[i]] += v[i].weight;
      }

#ifdef _REPORT_DETAILS
      cout << "[INFO]: distribution of nonzeros before balancing: ";
      for (int c = 0; c < ncolors_; c++)
        cout << ((float)load_nnz[c] /
                 (hybrid_ ? (nnz_hbw_ + nnz_low_) : nnz_low_)) *
                    100
             << " % ";
      cout << endl;
#endif
    }

    // Balance only first two color classes that contain most of the work
    int k_color = ncolors_;
    #pragma omp parallel num_threads(nthreads_)
    {
      int tid = omp_get_thread_num();
      SymThreadData *data = sym_thread_data_[tid];
      int total_load = 0;
      int mean_load = 0;
      vector<int> balance_deviation(k_color, 0);
      vector<int> load(ncolors_);

      if (part_by_ncnfls_) {
        int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
        for (int i = 0; i < nrows; ++i) {
          int row = data->rowind_[i << BlkBits] >> BlkBits;
          if (color[row] < k_color) {
            total_load += v[row].weight;
          }
          load[color[row]] += v[row].weight;
        }
      } else {
        int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
        int row_offset = row_split_[tid];
        for (int i = 0; i < nrows; ++i) {
          int row = ((i << BlkBits) + row_offset) >> BlkBits;
          if (color[row] < k_color) {
            total_load += v[row].weight;
          }
          load[color[row]] += v[row].weight;
        }
      }
      mean_load = total_load / k_color;

#ifdef _REPORT_DETAILS
      #pragma omp critical
      {
        cout << fixed;
        cout << setprecision(2);
        cout << "[INFO]: T" << tid
             << " load distribution before balancing = { ";
        for (int c = 0; c < k_color; c++) {
          cout << ((float)load[c] / total_load) * 100 << "% ";
        }
        cout << "}" << endl;
      }
#endif

      vector<queue<WeightedVertex>> bin(ncolors_); // first-fit
      for (int step = 0; step < ncolors_; ++step) {
        // Find total weight and vertices per color, total weight over all
        // colors
        if (part_by_ncnfls_) {
          int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
          for (int i = 0; i < nrows; ++i) {
            int row = data->rowind_[i << BlkBits] >> BlkBits;
            bin[color[row]].push(v[row]);
          }
        } else {
          int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
          int row_offset = row_split_[tid];
          for (int i = 0; i < nrows; ++i) {
            int row = ((i << BlkBits) + row_offset) >> BlkBits;
            bin[color[row]].push(v[row]);
          }
        }

        // Find balance deviation for this processor
        for (int c = 0; c < k_color; c++) {
          balance_deviation[c] = load[c] - mean_load;
        }

        // Minimize balance deviation of each color c
        // The deviance reduction heuristic works by moving vertices from one
        // color with positive deviation to another legal color with a lower
        // deviation when this exchange will reduce the total deviation.
        // This is similar to a bin-packing problem, with the added constraint
        // that a vertex cannot be placed in the same bin as its neighbors.
        // Find color with largest positive deviation
        int max_c = distance(
            balance_deviation.begin(),
            max_element(balance_deviation.begin(), balance_deviation.end()));
        while (load[max_c] - mean_load > ImbalanceTol && !bin[max_c].empty()) {
          WeightedVertex current = bin[max_c].front();
          // Find eligible colors for this vertex
          vector<bool> used(ncolors_, false);
          used[max_c] = true;
          const auto &neighbors = g[current.vid];
          for (auto j : neighbors)
            used[color[j]] = true;

          int target_c = (max_c + 1) % ncolors_;
          if (!used[target_c] && target_c != max_c) {
            // Update color bins
            color[current.vid] = target_c;
            load[max_c] -= current.weight;
            load[target_c] += current.weight;
            bin[target_c].push(current);
          }
          bin[max_c].pop();
        }
      }

#ifdef _REPORT_DETAILS
      #pragma omp critical
      {
        cout << "[INFO]: T" << tid << " load distribution after balancing = { ";
        for (int c = 0; c < k_color; c++) {
          cout << ((float)load[c] / total_load) * 100 << "% ";
        }
        cout << "}" << endl;
      }
#endif
    }

#ifdef _REPORT_DETAILS
    vector<int> load_nnz2(ncolors_, 0);
    if (!v.empty()) {
      for (int i = 0; i < V; ++i) {
        load_nnz2[color[i]] += v[i].weight;
      }

      cout << "[INFO]: distribution of nonzeros after balancing: ";
      for (int c = 0; c < ncolors_; c++)
        cout << ((float)load_nnz2[c] /
                 (hybrid_ ? (nnz_hbw_ + nnz_low_) : nnz_low_)) *
                    100
             << " % ";
      cout << endl;
    }
#endif
  }

#ifdef _LOG_INFO
  float color_stop = omp_get_wtime();
  cout << "[INFO]: graph coloring: " << color_stop - color_start << endl;
  cout << "[INFO]: found " << ncolors_ << " colors" << endl;
#endif
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::color_greedy(const ConflictGraph &g,
                                             const vector<WeightedVertex> &v,
                                             bool balance, vector<int> &color) {
  assert(symmetric_ && cmp_symmetry_);
#ifdef _LOG_INFO
  cout << "[INFO]: applying distance-1 graph coloring to detect "
          "conflict-free submatrices"
       << endl;
  float color_start = omp_get_wtime();
#endif

#ifdef _USE_ORDERING
  // Modify vertex ordering to improve coloring
  vector<int> order;
  ordering_heuristic(g, order);
#endif

  const int V = g.size();
  int max_color = 0;

  // We need to keep track of which colors are used by
  // adjacent vertices. We do this by marking the colors
  // that are used. The mark array contains the mark
  // for each color. The length of mark is the
  // number of vertices since the maximum possible number of colors
  // is the number of vertices.
  vector<int> mark(V, numeric_limits<int>::max());

  // Determine the color for every vertex one by one
  for (int i = 0; i < V; i++) {
#ifdef _USE_ORDERING
    const auto &neighbors = g[order[i]];
#else
    const auto &neighbors = g[i];
#endif

    // Mark the colors of vertices adjacent to current.
    // i can be the value for marking since i increases successively
    for (auto j : neighbors)
      mark[color[j]] = i;

    // Next step is to assign the smallest un-marked color
    // to the current vertex.
    int j = 0;

    // Scan through all useable colors, find the smallest possible
    // color that is not used by neighbors. Note that if mark[j]
    // is equal to i, color j is used by one of the current vertex's
    // neighbors.
    while (j < max_color && mark[j] == i)
      ++j;

    if (j == max_color) // All colors are used up. Add one more color
      ++max_color;

#ifdef _USE_ORDERING
    // At this point, j is the smallest possible color
    color[order[i]] = j;
#else
    // At this point, j is the smallest possible color
    color[i] = j;
#endif
  }

  ncolors_ = max_color;
  for (int i = 0; i < V; i++) {
    assert(color[i] < ncolors_);
  }

  // Balancing phase
  if (balance) {
    // Find color class sizes in number of nonzeros (histogram
    // computation)
    vector<int> load_nnz(ncolors_, 0);
    if (!v.empty()) {
      for (int i = 0; i < V; ++i) {
        load_nnz[color[i]] += v[i].weight;
      }

#ifdef _REPORT_DETAILS
      cout << "[INFO]: distribution of nonzeros before balancing: ";
      for (int c = 0; c < ncolors_; c++)
        cout << ((float)load_nnz[c] /
                 (hybrid_ ? (nnz_hbw_ + nnz_low_) : nnz_low_)) *
                    100
             << " % ";
      cout << endl;
#endif
    }

    // Balance only first two color classes that contain most of the work
    int k_color = 2;
    #pragma omp parallel num_threads(nthreads_)
    {
      int tid = omp_get_thread_num();
      SymThreadData *data = sym_thread_data_[tid];
      int total_load = 0;
      int mean_load = 0;
      vector<int> balance_deviation(k_color, 0);
      vector<int> load(ncolors_);

      if (part_by_ncnfls_) {
        int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
        for (int i = 0; i < nrows; ++i) {
          int row = data->rowind_[i << BlkBits] >> BlkBits;
          if (color[row] < k_color) {
            total_load += v[row].weight;
          }
          load[color[row]] += v[row].weight;
        }
      } else {
        int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
        int row_offset = row_split_[tid];
        for (int i = 0; i < nrows; ++i) {
          int row = ((i << BlkBits) + row_offset) >> BlkBits;
          if (color[row] < k_color) {
            total_load += v[row].weight;
          }
          load[color[row]] += v[row].weight;
        }
      }
      mean_load = total_load / k_color;

#ifdef _REPORT_DETAILS
      #pragma omp critical
      {
        cout << fixed;
        cout << setprecision(2);
        cout << "[INFO]: T" << tid
             << " load distribution before balancing = { ";
        for (int c = 0; c < k_color; c++) {
          cout << ((float)load[c] / total_load) * 100 << "% ";
        }
        cout << "}" << endl;
      }
#endif

      for (int step = 0; step < ncolors_ - 1; ++step) {
        vector<queue<WeightedVertex>> bin(ncolors_); // first-fit
        // Find total weight and vertices per color, total weight over all
        // colors
        if (part_by_ncnfls_) {
          int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
          for (int i = 0; i < nrows; ++i) {
            int row = data->rowind_[i << BlkBits] >> BlkBits;
            bin[color[row]].push(v[row]);
          }
        } else {
          int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
          int row_offset = row_split_[tid];
          for (int i = 0; i < nrows; ++i) {
            int row = ((i << BlkBits) + row_offset) >> BlkBits;
            bin[color[row]].push(v[row]);
          }
        }

        // Find balance deviation for this processor
        for (int c = 0; c < k_color; c++) {
          balance_deviation[c] = load[c] - mean_load;
        }

        // Minimize balance deviation of each color c
        // The deviance reduction heuristic works by moving vertices from one
        // color with positive deviation to another legal color with a lower
        // deviation when this exchange will reduce the total deviation.
        // This is similar to a bin-packing problem, with the added constraint
        // that a vertex cannot be placed in the same bin as its neighbors.
        // Find color with largest positive deviation
        int max_c = distance(
            balance_deviation.begin(),
            max_element(balance_deviation.begin(), balance_deviation.end()));
        while (load[max_c] - mean_load > ImbalanceTol && !bin[max_c].empty()) {
          WeightedVertex current = bin[max_c].front();
          // Find eligible colors for this vertex
          vector<bool> used(ncolors_, false);
          used[max_c] = true;
          const auto &neighbors = g[current.vid];
          for (auto j : neighbors)
            used[color[j]] = true;

          int target_c = (max_c + 1) % k_color; // FIXME
          if (!used[target_c] && target_c != max_c) {
            // Update color bins
            color[current.vid] = target_c;
            load[max_c] -= current.weight;
            load[target_c] += current.weight;
            bin[target_c].push(current);
          }
          bin[max_c].pop();
        }
      }

#ifdef _REPORT_DETAILS
      #pragma omp critical
      {
        cout << "[INFO]: T" << tid << " load distribution after balancing = { ";
        for (int c = 0; c < k_color; c++) {
          cout << ((float)load[c] / total_load) * 100 << "% ";
        }
        cout << "}" << endl;
      }
#endif
    }
  }

#ifdef _REPORT_DETAILS
  vector<int> load_nnz2(max_color, 0);
  if (!v.empty()) {
    for (int i = 0; i < V; ++i) {
      load_nnz2[color[i]] += v[i].weight;
    }

    cout << "[INFO]: distribution of nonzeros after balancing: ";
    for (int c = 0; c < max_color; c++)
      cout << ((float)load_nnz2[c] /
               (hybrid_ ? (nnz_hbw_ + nnz_low_) : nnz_low_)) *
                  100
           << " % ";
    cout << endl;
  }
#endif

// Deviance reduction balancing algorithm
//   if (balance) {
//     #pragma omp parallel num_threads(nthreads_)
//     {
//       int tid = omp_get_thread_num();
//       SymThreadData *data = sym_thread_data_[tid];

//       for (int step = 0; step < BalancingSteps; ++step) {
// 	int total_load = 0;
// 	int mean_load = 0;
// 	vector<int> balance_deviation(max_color, 0);
// 	vector<int> load(max_color);
// 	vector<priority_queue<WeightedVertex, vector<WeightedVertex>,
// DecreasingVertexWeight>> bin(max_color);

// 	// Find total weight and vertices per color, total weight over all
// colors
// 	if (part_by_ncnfls_) {
// 	  int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
// 	  for (int i = 0; i < nrows; ++i) {
// 	    int row = data->rowind_[i << BlkBits] >> BlkBits;
// 	    total_load += v[row].weight;
// 	    load[color[row]] += v[row].weight;
// 	    bin[color[row]].push(v[row]);
// 	  }
// 	} else {
// 	  int nrows = (int)ceil(data->nrows_ / (double)BlkFactor);
// 	  int row_offset = row_split_[tid];
// 	  for (int i = 0; i < nrows; ++i) {
// 	    int row = ((i << BlkBits) + row_offset) >> BlkBits;
// 	    total_load += v[row].weight;
// 	    load[color[row]] += v[row].weight;
// 	    bin[color[row]].push(v[row]);
// 	  }
// 	}
// 	mean_load = total_load / max_color;

// #ifdef _LOG_INFO
// #pragma omp critical
//   	{
//   	  if (step == 0) {
//   	    cout << fixed;
//   	    cout << setprecision(2);
//   	    cout << "[INFO]: T" << tid
//   		 << " load distribution before balancing = { ";
//   	    for (int c = 0; c < max_color; c++) {
//   	      cout << ((float)load[c] / total_load) * 100 << "% ";
//   	    }
//   	    cout << "}" << endl;
//   	  }
//   	}
// #endif

//   	// Find balance deviation for this processor
//   	for (int c = 0; c < max_color; c++) {
//   	  balance_deviation[c] = load[c] - mean_load;
//   	}

//   	// Minimize balance deviation of each color c
//   	// The deviance reduction heuristic works by moving vertices from one
//   	// color with positive deviation to another legal color with a lower
//   	// deviation when this exchange will reduce the total deviation.
//   	// This is similar to a bin-packing problem, with the added constraint
//   	// that a vertex cannot be placed in the same bin as its neighbors.
//   	// Find color with largest positive deviation
//   	int max_c = distance(balance_deviation.begin(),
// 			     max_element(balance_deviation.begin(),
// balance_deviation.end()));
//   	// FIXME check that this is a different max_c from previous step
//   	while (load[max_c] - mean_load > ImbalanceTol && !bin[max_c].empty()) {
//   	  WeightedVertex current = bin[max_c].top();
//   	  // Find eligible colors for this vertex
//   	  vector<bool> used(max_color, false);
//   	  used[max_c] = true;
//   	  const auto &neighbors = g[current.vid];
//   	  for (auto j : neighbors)
//   	    used[color[j]] = true;

//   	  int target_c = max_c;
//   	  // 1. Recolor with the first permissible color (FF)
//   	  // for (int c = 0; c < max_color; c++) {
//   	  //   if (!used[c]) {
//   	  //     target_c = c;
//   	  //     break;
//   	  //   }
//   	  // }
//   	  // 2. Recolor with the least-used (under-full) permissible color (LU)
//   	  int min_load = load[max_c];
//   	  for (int c = 0; c < max_color; c++) {
//   	    if (!used[c] && load[c] < min_load) {
//   	      target_c = c;
//   	      min_load = load[c];
//   	    }
//   	  }

//   	  if (target_c != max_c) {
//   	    // Update color bins
//   	    color[current.vid] = target_c;
//   	    load[max_c] -= current.weight;
//   	    load[target_c] += current.weight;
//   	    bin[target_c].push(current);
//   	  }

//   	  // Remove current row from candidates
//   	  bin[max_c].pop();
//   	}

// 	#ifdef _LOG_INFO
// #pragma omp critical
//   	{
//   	  if (step == BalancingSteps - 1) {
//   	    cout << fixed;
//   	    cout << setprecision(2);
//   	    cout << "[INFO]: T" << tid
//   		 << " load distribution after balancing = { ";
//   	    for (int c = 0; c < max_color; c++) {
//   	      cout << ((float)load[c] / total_load) * 100 << "% ";
//   	    }
//   	    cout << "}" << endl;
//   	  }
//   	}
// #endif
//       }
//     }
//   }

#ifdef _LOG_INFO
  float color_stop = omp_get_wtime();
  cout << "[INFO]: graph coloring: " << color_stop - color_start << endl;
  cout << "[INFO]: found " << ncolors_ << " colors" << endl;
#endif
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::parallel_color(
    const ConflictGraph &g, tbb::concurrent_vector<int> &color) {
  assert(symmetric_ && cmp_symmetry_);
#ifdef _LOG_INFO
  cout << "[INFO]: applying distance-1 parallel graph coloring to detect "
          "conflict-free submatrices"
       << endl;
  float color_start = omp_get_wtime();
#endif

#ifdef _USE_ORDERING
  // Modify vertex ordering to improve coloring
  vector<int> order;
  ordering_heuristic(g, order);
#endif

  const int V = g.size();
  vector<int> uncolored(V);

  #pragma omp parallel for schedule(static) num_threads(nthreads_)
  for (int i = 0; i < V; i++) {
#ifdef _USE_ORDERING
    uncolored[i] = order[i];
#else
    uncolored[i] = i;
#endif
  }

  int max_color_global = 0;
  int max_color[nthreads_] = {0};
  int U = V;
  while (U > 0) {
    // Phase 1: tentative coloring (parallel)
    #pragma omp parallel num_threads(nthreads_)
    {
      vector<int> mark(V, numeric_limits<int>::max());
      #pragma omp for schedule(static)
      for (int i = 0; i < U; i++) {
        int tid = omp_get_thread_num();
        int current = uncolored[i];
        const auto &neighbors = g[current];

        // We need to keep track of which colors are used by neightbors.
        // do this by marking the colors that are used.
        // unordered_map<int, int> mark;
        for (auto j : neighbors)
          mark[color[j]] = i;

        // Next step is to assign the smallest un-marked color to the current
        // vertex.
        int j = 0;

        // Find the smallest possible color that is not used by neighbors.
        while (j < max_color[tid] && mark[j] == i)
          ++j;

        // All colors are used up. Add one more color.
        if (j == max_color[tid])
          ++max_color[tid];

        // At this point, j is the smallest possible color. Save the color of
        // vertex current.
        color[current] = j; // Save the color of vertex current
      }

      #pragma omp barrier
      #pragma omp single
      for (int i = 0; i < nthreads_; i++) {
        if (max_color[i] > max_color_global) {
          max_color_global = max_color[i];
        }
        max_color[i] = max_color_global;
      }

      // Phase 2: conflict detection (parallel)
      #pragma omp for schedule(static)
      for (int i = 0; i < U; i++) {
        int current = uncolored[i];
        const auto &neighbors = g[current];
        for (auto j : neighbors) {
          // Add higher numbered vertex to uncolored set
          if ((color[j] == color[current])) {
            color[current] = V - 1;
          }
        }
      }
    }

    int tail = 0;
    for (int i = 0; i < U; i++) {
      if (color[uncolored[i]] == V - 1) {
        uncolored[tail++] = uncolored[i];
      }
    }

    U = tail;
  }

  ncolors_ = max_color_global;

#ifdef _LOG_INFO
  float color_stop = omp_get_wtime();
  cout << "[INFO]: graph coloring: " << color_stop - color_start << endl;
  cout << "[INFO]: using " << ncolors_ << " colors" << endl;
#endif
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::parallel_color(
    const ConflictGraph &g, const vector<WeightedVertex> &v,
    tbb::concurrent_vector<int> &color) {
  assert(symmetric_ && cmp_symmetry_);
#ifdef _LOG_INFO
  cout << "[INFO]: applying distance-1 parallel balanced graph coloring to "
          "detect conflict-free submatrices"
       << endl;
  float color_start = omp_get_wtime();
#endif

  // Modify vertex ordering to improve coloring
  vector<int> order;
  ordering_heuristic(g, order);

  const int V = g.size();
  vector<int> uncolored(V);

  #pragma omp parallel for schedule(static) num_threads(nthreads_)
  for (int i = 0; i < V; i++) {
#ifdef _USE_ORDERING
    uncolored[i] = order[i];
#else
    uncolored[i] = i;
#endif
  }

  int max_color_global = 0;
  int max_color[nthreads_] = {0};
  int U = V;
  while (U > 0) {
    // Phase 1: tentative coloring (parallel)
    #pragma omp parallel num_threads(nthreads_)
    {
      vector<int> mark(V, numeric_limits<int>::max());
      #pragma omp for schedule(static)
      for (int i = 0; i < U; i++) {
        int tid = omp_get_thread_num();
        int current = uncolored[i];
        const auto &neighbors = g[current];

        // We need to keep track of which colors are used by neightbors.
        // do this by marking the colors that are used.
        // unordered_map<int, int> mark;
        for (auto j : neighbors)
          mark[color[j]] = i;

        // Next step is to assign the smallest un-marked color to the current
        // vertex.
        int j = 0;

        // Find the smallest possible color that is not used by neighbors.
        while (j < max_color[tid] && mark[j] == i)
          ++j;

        // All colors are used up. Add one more color.
        if (j == max_color[tid])
          ++max_color[tid];

        // At this point, j is the smallest possible color. Save the color of
        // vertex current.
        color[current] = j; // Save the color of vertex current
      }
    }

    for (int i = 0; i < nthreads_; i++) {
      if (max_color[i] > max_color_global) {
        max_color_global = max_color[i];
      }
      max_color[i] = max_color_global;
    }

    // Phase 2: conflict detection (parallel)
    #pragma omp parallel for schedule(static) num_threads(nthreads_)
    for (int i = 0; i < U; i++) {
      int current = uncolored[i];
      const auto &neighbors = g[current];
      for (auto j : neighbors) {
        // Add higher numbered vertex to uncolored set
        if ((color[j] == color[current])) {
          color[current] = V - 1;
        }
      }
    }

    int tail = 0;
    for (int i = 0; i < U; i++) {
      if (color[uncolored[i]] == V - 1) {
        uncolored[tail++] = uncolored[i];
      }
    }

    U = tail;
  }

  ncolors_ = max_color_global;

  // Phase 3: color balancing
  for (int t = 0; t < nthreads_; t++) {
    int total_load = 0;
    int mean_load = 0;
    vector<int> load(ncolors_);
    int balance_deviation[ncolors_] = {0};
    vector<vector<WeightedVertex>> bin(ncolors_);

    // Find total weight and vertices per color, total weight over all colors
    // and balance deviation for this processor
    for (int i = 0; i < V; i++) {
      if (v[i].tid == t) {
        total_load += v[i].weight;
        load[color[i]] += v[i].weight;
        bin[color[i]].push_back(v[i]);
      }
    }
    mean_load = total_load / ncolors_;
    for (int c = 0; c < ncolors_; c++) {
      balance_deviation[c] = load[c] - mean_load;
    }

// Sort vertices per color bin in descending order of nonzeros
// for (int c = 0; c < max_color; c++) {
//   sort(bin[c].begin(), bin[c].end());
// }

#ifdef _LOG_INFO
    cout << fixed;
    cout << setprecision(2);
    cout << "[INFO]: T" << t << " load distribution before balancing = { ";
    for (int c = 0; c < ncolors_; c++) {
      cout << ((float)load[c] / total_load) * 100 << "% ";
    }
    cout << "}" << endl;
#endif

    // Minimize balance deviation of each color c
    // The deviance reduction heuristic works by moving vertices from one color
    // with positive deviation to another legal color with a lower deviation
    // when this exchange with reduce the total deviation.
    // This is similar to a bin-packing problem, with the added constraint that
    // a vertex cannot be placed in the same bin as its neighbors.
    // Find color with largest positive deviation
    // unsigned int max_deviation = *max_element(balance_deviation,
    // balance_deviation + ncolors_);
    int max_c =
        distance(balance_deviation,
                 max_element(balance_deviation, balance_deviation + ncolors_));
    int i = 0;
    while (abs(load[max_c] - mean_load) > ImbalanceTol && !bin[max_c].empty()) {
      int current = bin[max_c][i].vid;
      // Find eligible colors for this vertex
      bool used[ncolors_] = {false};
      used[max_c] = true;
      const auto &neighbors = g[current];
      for (auto j : neighbors) {
        assert(color[j] < ncolors_);
        used[color[j]] = true;
      }

      // Re-color with the smallest eligible bin
      int min_c = max_c;
      int min_load = load[max_c];
      for (int c = 0; c < ncolors_; c++) {
        if (!used[c] && load[c] < min_load) {
          min_c = c;
          min_load = load[c];
        }
      }
      color[current] = min_c;
      load[max_c] -= v[current].weight;
      load[min_c] += v[current].weight;

      i++;
    }

#ifdef _LOG_INFO
    cout << "[INFO]: T" << t << " load distribution after balancing = { ";
    for (int c = 0; c < ncolors_; c++) {
      cout << ((float)load[c] / total_load) * 100 << "% ";
    }
    cout << "}" << endl;
#endif
  }

#ifdef _LOG_INFO
  float color_stop = omp_get_wtime();
  cout << "[INFO]: graph coloring: " << color_stop - color_start << endl;
  cout << "[INFO]: using " << ncolors_ << " colors" << endl;
#endif
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_serial(ValueT *__restrict y,
                                              const ValueT *__restrict x) {
  for (IndexT i = 0; i < nrows_; ++i) {
    register ValueT y_tmp = 0;

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

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv(ValueT *__restrict y,
                                       const ValueT *__restrict x) {
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    for (IndexT i = row_split_[tid]; i < row_split_[tid + 1]; ++i) {
      register ValueT y_tmp = 0;

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

  SymThreadData *data = sym_thread_data_[0];
  IndexT *rowptr = data->rowptr_;
  IndexT *colind = data->colind_;
  ValueT *values = data->values_;
  ValueT *diagonal = data->diagonal_;

  for (int i = 0; i < nrows_; ++i) {
    ValueT y_tmp = diagonal[i] * x[i];

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
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_atomics(ValueT *__restrict y,
                                                   const ValueT *__restrict x) {

  // Local vectors phase
  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymThreadData *data = sym_thread_data_[tid];
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

      for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        IndexT col = colind[j];
        ValueT val = values[j];
        y_tmp += val * x[col];
        #pragma omp atomic
        y[col] += val * x[i + row_offset];
      }

      #pragma omp atomic
      y[i + row_offset] += y_tmp;
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_effective_ranges(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymThreadData *data = sym_thread_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    ValueT *y_local = data->local_vector_;
    if (tid == 0)
      y_local = y;

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    // Local vectors phase
    for (int i = 0; i < data->nrows_; ++i) {
      register ValueT y_tmp = 0;

      for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        IndexT col = colind[j];
        ValueT val = values[j];
        y_tmp += val * x[col];
        if (col < row_offset)
          y_local[col] += val * x[i + row_offset];
        else
          y[col] += val * x[i + row_offset];
      }

      /* Reduction on y */
      y[i + row_offset] += y_tmp;
    }
    #pragma omp barrier

    // Reduction of conflicts phase
    for (int tid = 1; tid < nthreads_; ++tid) {
      SymThreadData *data = sym_thread_data_[tid];
      ValueT *y_local = data->local_vector_;
      #pragma omp for schedule(static)
      for (IndexT i = 0; i < row_split_[tid]; ++i) {
        y[i] += y_local[i];
        y_local[i] = 0;
      }
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_local_vectors_indexing(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    IndexT row_offset = row_split_[tid];
    SymThreadData *data = sym_thread_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    ValueT *y_local = data->local_vector_;
    if (tid == 0) {
      y_local = y;
    }

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    // Local vectors phase
    for (int i = 0; i < data->nrows_; ++i) {
      register ValueT y_tmp = 0;

      for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        IndexT col = colind[j];
        ValueT val = values[j];
        y_tmp += val * x[col];
        if (col < row_offset)
          y_local[col] += val * x[i + row_offset];
        else
          y[col] += val * x[i + row_offset];
      }

      /* Reduction on y */
      y[i + row_offset] += y_tmp;
    }
    #pragma omp barrier

    for (int i = data->map_start_; i < data->map_end_; ++i) {
      y[cnfl_map_->pos[i]] += y_local_[cnfl_map_->cpu[i]][cnfl_map_->pos[i]];
      y_local_[cnfl_map_->cpu[i]][cnfl_map_->pos[i]] = 0.0;
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_apriori(
    ValueT *__restrict y, const ValueT *__restrict x) {

  SymThreadData *data = sym_thread_data_[0];
  IndexT *rowptr = data->rowptr_;
  IndexT *rowind = data->rowind_;
  IndexT *colind = data->colind_;
  ValueT *values = data->values_;
  ValueT *diagonal = data->diagonal_;
  IndexT *color_ptr = data->color_ptr_;

  #pragma omp parallel num_threads(nthreads_)
  {
    for (int c = 0; c < ncolors_; ++c) {
      #pragma omp for schedule(static, BlkFactor * 64)
      for (int i = color_ptr[c]; i < color_ptr[c + 1]; ++i) {
        register IndexT row = rowind[i];
        register ValueT y_tmp = diagonal[row] * x[row];

        for (IndexT j = rowptr[row]; j < rowptr[row + 1]; ++j) {
          IndexT col = colind[j];
          ValueT val = values[j];
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
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_v1(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    SymThreadData *data = sym_thread_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *rowind = data->rowind_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;

    for (int i = 0; i < data->nrows_; ++i) {
      IndexT row = rowind[i];
      y[row] = diagonal[i] * x[row];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
#ifndef _USE_BARRIER
      // Wait until my dependencies have finished the previous phase
      for (size_t i = 0; i < data->deps_[c].size(); i++)
        while (!done[data->deps_[c][i]][c - 1].load(memory_order_acquire))
          continue;
#endif
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          IndexT row = rowind[i];
          register ValueT y_tmp = 0;

          for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IndexT col = colind[j];
            ValueT val = values[j];
            y_tmp += val * x[col];
            y[col] += val * x[row];
          }

          /* Reduction on y */
          y[row] += y_tmp;
        }
      }

#ifdef _USE_BARRIER
#pragma omp barrier
#else
      // Inform threads that depend on me that I have completed this phase
      // Release-acquire synchronization is used to make sure that, once the
      // atomic load of threads waiting on me is completed, they are guaranteed
      // to see everything I wrote to memory.
      done[tid][c].store(true, memory_order_release);
#endif
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_v2(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    SymThreadData *data = sym_thread_data_[tid];
    IndexT row_offset = data->row_offset_;
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;

#ifndef _USE_BARRIER
    for (int c = 0; c < ncolors_; ++c) {
      util::runtime::done[tid][c].store(false);
    }
#endif

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
#ifndef _USE_BARRIER
      // Wait until my dependencies have finished the previous phase
      for (size_t i = 0; i < data->deps_[c].size(); i++)
        while (!util::runtime::done[data->deps_[c][i]][c - 1].load(
            memory_order_acquire))
          continue;
#endif
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          register ValueT y_tmp = 0;

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

#ifdef _USE_BARRIER
#pragma omp barrier
#else
      // Inform threads that depend on me that I have completed this phase
      // Release-acquire synchronization is used to make sure that, once the
      // atomic load of threads waiting on me is completed, they are guaranteed
      // to see everything I wrote to memory.
      util::runtime::done[tid][c].store(true, memory_order_release);
#endif
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_hyb_v1(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    SymThreadData *data = sym_thread_data_[tid];
    IndexT *rowptr = data->rowptr_;
    IndexT *rowind = data->rowind_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;
    IndexT *rowptr_h = data->rowptr_h_;
    IndexT *colind_h = data->colind_h_;
    ValueT *values_h = data->values_h_;

    for (int i = 0; i < data->nrows_; ++i) {
      IndexT row = rowind[i];
      y[row] = diagonal[i] * x[row];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
#ifndef _USE_BARRIER
      // Wait until my dependencies have finished the previous phase
      for (size_t i = 0; i < data->deps_[c].size(); i++)
        while (!util::runtime::done[data->deps_[c][i]][c - 1].load(
            memory_order_relaxed))
          continue;
#endif
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          IndexT row = rowind[i];
          register ValueT y_tmp = 0;

          for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IndexT col = colind[j];
            ValueT val = values[j];
            y_tmp += val * x[col];
            y[col] += val * x[row];
          }

          for (IndexT j = rowptr_h[i]; j < rowptr_h[i + 1]; ++j) {
            y_tmp += values_h[j] * x[colind_h[j]];
          }

          /* Reduction on y */
          y[row] += y_tmp;
        }
      }

#ifdef _USE_BARRIER
#pragma omp barrier
#else
      // Inform threads that depend on me that I have completed this phase
      util::runtime::done[tid][c].store(true);
#endif
    }
  }
}

template <typename IndexT, typename ValueT>
void CSRMatrix<IndexT, ValueT>::cpu_mv_sym_conflict_free_hyb_v2(
    ValueT *__restrict y, const ValueT *__restrict x) {

  #pragma omp parallel num_threads(nthreads_)
  {
    int tid = omp_get_thread_num();
    SymThreadData *data = sym_thread_data_[tid];
    IndexT row_offset = data->row_offset_;
    IndexT *rowptr = data->rowptr_;
    IndexT *colind = data->colind_;
    ValueT *values = data->values_;
    ValueT *diagonal = data->diagonal_;
    IndexT *range_ptr = data->range_ptr_;
    IndexT *range_start = data->range_start_;
    IndexT *range_end = data->range_end_;
    IndexT *rowptr_h = data->rowptr_h_;
    IndexT *colind_h = data->colind_h_;
    ValueT *values_h = data->values_h_;

#ifndef _USE_BARRIER
    for (int c = 0; c < ncolors_; ++c) {
      util::runtime::done[tid][c].store(false);
    }
#endif

    for (int i = 0; i < data->nrows_; ++i) {
      y[i + row_offset] = diagonal[i] * x[i + row_offset];
    }
    #pragma omp barrier

    for (int c = 0; c < ncolors_; ++c) {
#ifndef _USE_BARRIER
      // Wait until my dependencies have finished the previous phase
      for (size_t i = 0; i < data->deps_[c].size(); i++)
        while (!util::runtime::done[data->deps_[c][i]][c - 1].load(
            memory_order_relaxed))
          continue;
#endif
      for (int r = range_ptr[c]; r < range_ptr[c + 1]; ++r) {
        for (IndexT i = range_start[r]; i <= range_end[r]; ++i) {
          register ValueT y_tmp = 0;

          for (IndexT j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            IndexT col = colind[j];
            ValueT val = values[j];
            y_tmp += val * x[col];
            y[col] += val * x[i + row_offset];
          }

          for (IndexT j = rowptr_h[i]; j < rowptr_h[i + 1]; ++j) {
            y_tmp += values_h[j] * x[colind_h[j]];
          }

          /* Reduction on y */
          y[i + row_offset] += y_tmp;
        }
      }

#ifdef _USE_BARRIER
#pragma omp barrier
#else
      // Inform threads that depend on me that I have completed this phase
      util::runtime::done[tid][c].store(true);
#endif
    }
  }
}

} // end of namespace sparse
} // end of namespace matrix
} // end of namespace cfs

#endif
