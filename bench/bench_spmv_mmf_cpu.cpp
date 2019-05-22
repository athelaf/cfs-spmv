#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string.h>

#ifdef _INTEL_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

#ifdef _RSB
#include <rsb.h>
#include <blas_sparse.h>
#endif

#include "kernel/sparse_kernels.hpp"
#include "matrix/sparse_matrix.hpp"

using namespace std;
using namespace matrix::sparse;
using namespace kernel::sparse;

char *program_name = NULL;

typedef int INDEX;
#ifdef _USE_DOUBLE
typedef double VALUE;
#else
typedef float VALUE;
#endif

static void set_program_name(char *path) {
  if (!program_name)
    program_name = strdup(path);
  if (!program_name)
    fprintf(stderr, "strdup failed\n");
}

static void print_usage() {
  cout << "Usage: " << program_name
       << " <mmf_file> <format>(0: COO, 1: CSR, 2:SSS, 3: HYB, 4: MKL-CSR, 5: "
          "MKL-BSR, 6: RSB) <iterations>"
       << endl;
}

int main(int argc, char **argv) {
  set_program_name(argv[0]);
  if (argc < 4) {
    cerr << "Error in number of arguments!" << endl;
    print_usage();
    exit(1);
  }

  const string mmf_file(argv[1]);
  int format = atoi(argv[2]);
  if (format > 6) {
    cerr << "Error in arguments!" << endl;
    print_usage();
    exit(1);
  }
  size_t loops = atoi(argv[3]);

  // Load a sparse matrix from an MMF file
  SparseMatrix<INDEX, VALUE> *A = nullptr;
  A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::coo);
  int M = A->nrows();
  int N = A->ncols();
  int nnz = A->nnz();

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis_val(0.01, 0.42);

  VALUE *y = (VALUE *)internal_alloc(M * sizeof(VALUE));
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; i++) {
    y[i] = 0.0;
  }

  VALUE *x = (VALUE *)internal_alloc(N * sizeof(VALUE));
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    x[i] = dis_val(gen);
  }

  double compute_time = 0.0, preproc_time = 0.0, tstart = 0.0, tstop = 0.0,
         gflops = 0.0;
  string format_string;
  switch (format) {
  case 0: {
    format_string = "COO";
    break;
  }
  case 1: {
    A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::csr);
    format_string = "CSR";
    break;
  }
  case 2: {
    A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::sss);
    format_string = "SSS";
    break;
  }
  case 3: {
    A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::hyb);
    format_string = "HYB";
    break;
  }
  case 4: {
    format_string = "MKL-CSR";
    break;
  }
  case 5: {
    format_string = "MKL-BSR";
    break;
  }
  case 6: {
    format_string = "RSB";
    break;
  }
  }

  if (format < 4) {
    tstart = omp_get_wtime();
    SpDMV<INDEX, VALUE> spdmv(A);
    tstop = omp_get_wtime();
    preproc_time = tstop - tstart;

#ifdef _LOG_INFO
    cout << "[INFO]: warming up caches..." << endl;
#endif
    // Warm up run
    for (size_t i = 0; i < loops / 2; i++)
      spdmv(y, M, x, N);

#ifdef _LOG_INFO
    cout << "[INFO]: benchmarking SpDMV using " << format_string << "..."
         << endl;
#endif
    // Benchmark run
    tstart = omp_get_wtime();
    for (size_t i = 0; i < loops; i++)
      spdmv(y, M, x, N);
    tstop = omp_get_wtime();
    compute_time = tstop - tstart;
    gflops = ((double)loops * 2 * nnz * 1.e-9) / compute_time;
    cout << setprecision(4) << "matrix: " << basename(mmf_file.c_str())
         << " format: " << format_string << " preproc(sec): " << preproc_time
         << " t(sec): " << compute_time / loops << " gflops/s: " << gflops
         << " threads: " << get_threads()
         << " size(MB): " << A->size() / (float)(1024 * 1024) << endl;

    // Cleanup
    delete A;
  }

  if (format == 4) {
#ifdef _INTEL_MKL
    CSRMatrix<INDEX, VALUE> *A_csr = new CSRMatrix<INDEX, VALUE>(mmf_file);
    VALUE alpha = 1, beta = 0;
    sparse_matrix_t A_view;
    sparse_status_t stat;
#ifdef _USE_DOUBLE
    stat = mkl_sparse_d_create_csr(&A_view, SPARSE_INDEX_BASE_ZERO, M, N,
                                   A_csr->rowptr(), A_csr->rowptr() + 1,
                                   A_csr->colind(), A_csr->values());
#else
    stat = mkl_sparse_s_create_csr(&A_view, SPARSE_INDEX_BASE_ZERO, M, N,
                                   A_csr->rowptr(), A_csr->rowptr() + 1,
                                   A_csr->colind(), A_csr->values());
#endif
    struct matrix_descr matdescr;
    // matdescr.type = SPARSE_MATRIX_TYPE_GENERAL;
    matdescr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    matdescr.mode = SPARSE_FILL_MODE_LOWER;
    matdescr.diag = SPARSE_DIAG_NON_UNIT;
    tstart = omp_get_wtime();
    stat = mkl_sparse_set_mv_hint(A_view, SPARSE_OPERATION_NON_TRANSPOSE,
                                  matdescr, 100000);
    stat = mkl_sparse_set_memory_hint(A_view, SPARSE_MEMORY_AGGRESSIVE);
    stat = mkl_sparse_optimize(A_view);
    tstop = omp_get_wtime();
    preproc_time = tstop - tstart;
    if (stat != SPARSE_STATUS_SUCCESS) {
      cout << "[INFO]: MKL auto-tuning failed" << endl;
    }

    mkl_set_num_threads(get_threads());
#ifdef _LOG_INFO
    cout << "[INFO]: warming up caches..." << endl;
#endif
    // Warm up run
    for (size_t i = 0; i < loops / 2; i++)
#ifdef _USE_DOUBLE
      mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
                      x, beta, y);
#else
      mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
                      x, beta, y);
#endif

#ifdef _LOG_INFO
    cout << "[INFO]: benchmarking SpDMV using MKL-CSR..." << endl;
#endif
    tstart = omp_get_wtime();
    // Benchmark run
    for (size_t i = 0; i < loops; i++)
#ifdef _USE_DOUBLE
      mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
                      x, beta, y);
#else
      mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
                      x, beta, y);
#endif
    tstop = omp_get_wtime();
    compute_time = tstop - tstart;

    gflops = ((double)loops * 2 * nnz * 1.e-9) / compute_time;
    cout << setprecision(4) << "matrix: " << basename(mmf_file.c_str())
         << " format: " << format_string << " preproc(sec): " << preproc_time
         << " t(sec): " << compute_time / loops << " gflops/s: " << gflops
         << " threads: " << get_threads() << endl;

    mkl_sparse_destroy(A_view);
    delete A_csr;
#endif // _INTEL_MKL
  }

  if (format == 5) {
#ifdef _INTEL_MKL
    CSRMatrix<INDEX, VALUE> *A_csr = new CSRMatrix<INDEX, VALUE>(mmf_file);
    sparse_status_t stat;
    sparse_matrix_t A_view_csr;
#ifdef _USE_DOUBLE
    stat = mkl_sparse_d_create_csr(&A_view_csr, SPARSE_INDEX_BASE_ZERO, M, N,
                                   A_csr->rowptr(), A_csr->rowptr() + 1,
                                   A_csr->colind(), A_csr->values());
#else
    stat = mkl_sparse_s_create_csr(&A_view_csr, SPARSE_INDEX_BASE_ZERO, M, N,
                                   A_csr->rowptr(), A_csr->rowptr() + 1,
                                   A_csr->colind(), A_csr->values());
#endif
    if (stat != SPARSE_STATUS_SUCCESS) {
      cout << "[INFO]: MKL conversion to CSR failed" << endl;
      exit(1);
    }

    sparse_matrix_t A_view_bsr;
    const sparse_operation_t op = SPARSE_OPERATION_NON_TRANSPOSE;
    const int blk_size = 3;
    tstart = omp_get_wtime();
    stat = mkl_sparse_convert_bsr(A_view_csr, blk_size, SPARSE_LAYOUT_ROW_MAJOR,
                                  op, &A_view_bsr);
    tstop = omp_get_wtime();
    preproc_time = tstop - tstart;
    if (stat != SPARSE_STATUS_SUCCESS) {
      cout << "[INFO]: MKL conversion to BSR failed with " << stat << endl;
      exit(1);
    }

    // Cleanup
    mkl_sparse_destroy(A_view_csr);
    delete A_csr;

    struct matrix_descr matdescr;
    matdescr.type = SPARSE_MATRIX_TYPE_GENERAL;
    VALUE alpha = 1, beta = 0;
    mkl_set_num_threads(get_threads());
#ifdef _LOG_INFO
    cout << "[INFO]: warming up caches..." << endl;
#endif
    // Warm up run
    for (size_t i = 0; i < loops / 2; i++)
#ifdef _USE_DOUBLE
      mkl_sparse_d_mv(op, alpha, A_view_bsr, matdescr, x, beta, y);
#else
      mkl_sparse_s_mv(op, alpha, A_view_bsr, matdescr, x, beta, y);
#endif

#ifdef _LOG_INFO
    cout << "[INFO]: benchmarking SpDMV using MKL-BSR..." << endl;
#endif
    tstart = omp_get_wtime();
    // Benchmark run
    for (size_t i = 0; i < loops; i++)
#ifdef _USE_DOUBLE
      mkl_sparse_d_mv(op, alpha, A_view_bsr, matdescr, x, beta, y);
#else
      mkl_sparse_s_mv(op, alpha, A_view_bsr, matdescr, x, beta, y);
#endif
    tstop = omp_get_wtime();
    compute_time = tstop - tstart;

    gflops = ((double)loops * 2 * nnz * 1.e-9) / compute_time;
    cout << setprecision(4) << "matrix: " << basename(mmf_file.c_str())
         << " format: " << format_string << " preproc(sec): " << preproc_time
         << " t(sec): " << compute_time / loops << " gflops/s: " << gflops
         << " threads: " << get_threads() << endl;

    mkl_sparse_destroy(A_view_bsr);
#endif // _INTEL_MKL
  }

  if (format == 6) {
#ifdef _RSB
    blas_sparse_matrix A = blas_invalid_handle;
    rsb_type_t typecode = RSB_NUMERICAL_TYPE_DOUBLE;
    rsb_err_t errval = RSB_ERR_NO_ERROR;
    VALUE alpha = 1;

    // Initialize library
    if ((errval = rsb_lib_init(RSB_NULL_INIT_OPTIONS)) != RSB_ERR_NO_ERROR) {
      cout << "Error initializing the RSB library" << endl;
      exit(1);
    }

    // Load matrix
    A = rsb_load_spblas_matrix_file_as_matrix_market(mmf_file.c_str(),
                                                     typecode);
    if (A == blas_invalid_handle) {
      cout << "Error while loading matrix from file" << endl;
      exit(1);
    }

    // Autotune
    tstart = omp_get_wtime();
    BLAS_ussp(A, blas_rsb_autotune_next_operation);
    BLAS_dusmv(blas_no_trans, alpha, A, x, 1, y, 1);
    tstop = omp_get_wtime();
    preproc_time = tstop - tstart;

#ifdef _LOG_INFO
    cout << "[INFO]: benchmarking SpDMV using RSB..." << endl;
#endif
    // Benchmark run
    tstart = omp_get_wtime();
    for (size_t i = 0; i < loops; i++)
      BLAS_dusmv(blas_no_trans, alpha, A, x, 1, y, 1);
    tstop = omp_get_wtime();
    compute_time = tstop - tstart;

    gflops = ((double)loops * 2 * nnz * 1.e-9) / compute_time;
    cout << setprecision(4) << "matrix: " << basename(mmf_file.c_str())
         << " format: " << format_string << " preproc(sec): " << preproc_time
         << " t(sec): " << compute_time / loops << " gflops/s: " << gflops
         << " threads: " << get_threads() << endl;

    // Cleanup
    BLAS_usds(A);
    if ((errval = rsb_lib_exit(RSB_NULL_EXIT_OPTIONS)) != RSB_ERR_NO_ERROR) {
      cout << "Error finalizing the RSB library" << endl;
      exit(1);
    }

#endif
  }

  // Cleanup
  internal_free(x);
  internal_free(y);
  free(program_name);

  return 0;
}
