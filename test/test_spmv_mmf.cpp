#include <iomanip>
#include <iostream>
#include <string.h>

#include "kernel/sparse_kernels.hpp"
#include "matrix/sparse_matrix.hpp"

using namespace std;
using namespace util;
using namespace matrix::sparse;
using namespace kernel::sparse;

char *program_name = NULL;

typedef int INDEX;
typedef double VALUE;

static void set_program_name(char *path) {
  if (!program_name)
    program_name = strdup(path);
  if (!program_name)
    fprintf(stderr, "strdup failed\n");
}

static void print_usage() {
  cout << "Usage: " << program_name
       << " <mmf_file> <format>(0: COO, 1: CSR, 2: CFS-SSS, 3: CFH-SSS)"
       << endl;
}

int main(int argc, char **argv) {
  set_program_name(argv[0]);
  if (argc < 3) {
    cerr << "Error in number of arguments!" << endl;
    print_usage();
    exit(1);
  }

  const string mmf_file(argv[1]);
  int fmt = atoi(argv[2]);
  if (fmt > 3) {
    cerr << "Error in arguments!" << endl;
    print_usage();
    exit(1);
  }

  // Load a sparse matrix from an MMF file
  SparseMatrix<INDEX, VALUE> *A = nullptr;
  switch (fmt) {
  case 0:
    A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::coo);
    break;
  case 1:
    A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::csr);
    break;
  case 2:
    A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::sss);
    break;
  case 3:
    A = createSparseMatrix<INDEX, VALUE>(mmf_file, Format::hyb);
    break;
  default:
    break;
  }

  int M = A->nrows();
  int N = A->ncols();
#ifdef _LOG_INFO
  double sparsity = (1 - ((A->nnz() / (double)(M)) / N)) * 100;
  cout << "[INFO]: sparsity " << sparsity << " %" << endl;
#endif

  VALUE *x = (VALUE *)internal_alloc(N * sizeof(VALUE));
  VALUE *y = (VALUE *)internal_alloc(M * sizeof(VALUE));

  random_device
      rd; // Will be used to obtain a seed for the random number engine
  mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  uniform_real_distribution<> dis_val(10.01, 20.42);
  for (int i = 0; i < N; i++) {
    x[i] = dis_val(gen);
  }

  SpDMV<INDEX, VALUE> fn(A);
  fn(y, M, x, N);

  SparseMatrix<INDEX, VALUE> *A_test =
      createSparseMatrix<INDEX, VALUE>(mmf_file, Format::coo);
  SpDMV<INDEX, VALUE> test(A_test, Tuning::None);
  VALUE *y_test = (VALUE *)internal_alloc(M * sizeof(VALUE));
  test(y_test, M, x, N);

#ifdef _LOG_INFO
  cout << "[INFO]: checking result... ";
#endif
  bool passed = true;
  for (INDEX i = 0; i < M; i++) {
    if (fabs((VALUE)(y[i] - y_test[i]) / (VALUE)y[i]) > EPS) {
      cout << "element " << i << " differs: " << y[i] << " vs " << y_test[i]
           << endl;
      passed = false;
      break;
    }
    if (!passed)
      break;
  }

  if (passed)
    cout << "PASSED!" << endl;
  else
    cout << "FAILED!" << endl;

  // Cleanup
  delete A;
  delete A_test;
  internal_free(x);
  internal_free(y);
  internal_free(y_test);
  free(program_name);

  return 0;
}
