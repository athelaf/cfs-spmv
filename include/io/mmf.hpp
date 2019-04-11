#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace util {
namespace io {

template <typename IndexType, typename ValueType> struct Elem {
  IndexType row;
  IndexType col;
  ValueType val;
};

template <typename IndexType, typename ValueType> struct ElemSorter {
public:
  bool operator()(const Elem<IndexType, ValueType> &lhs,
                  const Elem<IndexType, ValueType> &rhs) const {
    if (lhs.row < rhs.row)
      return true;
    if (lhs.row > rhs.row)
      return false;

    if (lhs.col < rhs.col)
      return true;
    if (lhs.col > rhs.col)
      return false;

    return false;
  }
};

bool DoRead(std::ifstream &in, std::vector<std::string> &arguments);

template <typename IndexType, typename ValueType>
void ParseElement(std::vector<std::string> &arguments, IndexType &y,
                  IndexType &x, ValueType &v);

template <typename IndexType, typename ValueType> class MMF {
public:
  typedef IndexType idx_t;
  typedef ValueType val_t;

  MMF(const std::string &filename);

  ~MMF() {
    if (in.is_open())
      in.close();
  }

  IndexType GetNrRows() const { return nr_rows_; }

  IndexType GetNrCols() const { return nr_cols_; }

  IndexType GetNrNonzeros() const {
    if (symmetric_ || col_wise_)
      return matrix_.size();
    else
      return nr_nzeros_;
  }

  bool IsSymmetric() const { return symmetric_; }

  bool IsColWise() const { return col_wise_; }

  bool IsZeroBased() const { return zero_based_; }

  class iterator;
  iterator begin();
  iterator end();

private:
  IndexType nr_rows_, nr_cols_, nr_nzeros_;
  std::ifstream in;
  bool symmetric_, col_wise_, zero_based_;
  int file_mode_; // 0 for MMF files, 1 for regular files
  std::vector<Elem<IndexType, ValueType>> matrix_;

  enum MmfInfo {
    Banner,
    Matrix,
    Coordinate,
    Real,
    Double,
    Integer,
    General,
    Symmetric,
    Indexing0,
    Indexing1,
    ColumnWise,
    RowWise
  };

  void ParseMmfHeaderLine(std::vector<std::string> &arguments);
  void ParseMmfSizeLine(std::vector<std::string> &arguments);
  void DoLoadMmfMatrix();
  bool GetNext(IndexType &y, IndexType &x, ValueType &val);
};

template <typename IndexType, typename ValueType>
class MMF<IndexType, ValueType>::iterator
    : public std::iterator<std::forward_iterator_tag,
                           Elem<IndexType, ValueType>> {
public:
  iterator() {}

  iterator(MMF *mmf, size_t cnt) : mmf_(mmf), cnt_(cnt) {
    if (mmf_->symmetric_ || mmf_->col_wise_)
      return;

    // this is the initializer
    if (cnt_ == 0) {
      this->DoSet();
    }
  }

  bool operator==(const iterator &i) {
    return (mmf_ == i.mmf_) && (cnt_ == i.cnt_);
  }

  bool operator!=(const iterator &i) { return !(*this == i); }

  void operator++() {
    ++cnt_;
    if (mmf_->symmetric_ || mmf_->col_wise_) {
      return;
    }
    this->DoSet();
  }

  Elem<IndexType, ValueType> &operator*() {
    if (mmf_->symmetric_ || mmf_->col_wise_) {
      return mmf_->matrix_[cnt_];
    } else {
      if (!valid_) {
        std::cout << "Requesting dereference, but mmf ended\n";
        exit(1);
      }
      assert(valid_);
      return elem_;
    }
  }

private:
  void DoSet() { valid_ = mmf_->GetNext(elem_.row, elem_.col, elem_.val); }

  MMF *mmf_;
  size_t cnt_;
  Elem<IndexType, ValueType> elem_;
  bool valid_;
};

template <typename IndexType, typename ValueType>
typename MMF<IndexType, ValueType>::iterator
MMF<IndexType, ValueType>::begin() {
  return iterator(this, 0);
}

template <typename IndexType, typename ValueType>
typename MMF<IndexType, ValueType>::iterator MMF<IndexType, ValueType>::end() {
  if (this->symmetric_ || this->col_wise_) {
    return iterator(this, matrix_.size());
  } else {
    return iterator(this, nr_nzeros_);
  }
}

/*
 * Implementation of class MMF
 */
template <typename IndexType, typename ValueType>
MMF<IndexType, ValueType>::MMF(const std::string &filename)
    : nr_rows_(0), nr_cols_(0), nr_nzeros_(0), symmetric_(false),
      col_wise_(true), zero_based_(false), file_mode_(0) {
  try {
    in.open(filename);
    if (!in.is_open()) {
      throw std::ios_base::failure("");
    }
  } catch (std::ios_base::failure &e) {
    std::cout << "MMF file error.\n";
    exit(1);
  }
  std::vector<std::string> arguments;

  DoRead(in, arguments);
  ParseMmfHeaderLine(arguments);
  ParseMmfSizeLine(arguments);

  if (symmetric_ || col_wise_) {
    DoLoadMmfMatrix();
  }
}

template <typename IndexType, typename ValueType>
void MMF<IndexType, ValueType>::ParseMmfHeaderLine(
    std::vector<std::string> &arguments) {
  // Check if header line exists
  if (arguments[0].compare("%%MatrixMarket")) {
    if (arguments[0].length() > 2 && arguments[0][0] == '%' &&
        arguments[0][1] == '%') {
      // Header exists but is erroneous so exit
      std::cout << "invalid header line in MMF file.\n";
      exit(1);
    } else {
      // Parse as size line
      file_mode_ = 1;
      return;
    }
  }

  size_t length;
  if ((length = arguments.size()) < 5) {
    std::cout << "less arguments in header line of MMF file.\n";
    exit(1);
  }

  if (arguments[2].compare("coordinate")) {
    std::cout << "unsupported matrix format in header line of MMF file.\n";
    exit(1);
  }

  if (arguments[4].compare("general") == 0) {
    symmetric_ = false;
  } else if (arguments[4].compare("symmetric") == 0) {
    symmetric_ = true;
  } else {
    std::cout << "unsupported symmetry in header line of MMF file.\n";
    exit(1);
  }

  if (length > 5) {
    for (size_t i = 5; i < length; i++) {
      if (arguments[i].compare("base-0") == 0)
        zero_based_ = true;
      else if (arguments[i].compare("base-1") == 0)
        zero_based_ = false;
      else if (arguments[i].compare("column") == 0)
        col_wise_ = true;
      else if (arguments[i].compare("row") == 0)
        col_wise_ = false;
    }
  }
}

template <typename IndexType, typename ValueType>
void MMF<IndexType, ValueType>::ParseMmfSizeLine(
    std::vector<std::string> &arguments) {
  bool ignore_comments = false;

  if (file_mode_ && arguments[0][0] == '%') {
    ignore_comments = true;
  }

  if (!file_mode_ || ignore_comments) {
    while (in.peek() == '%') {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (!DoRead(in, arguments)) {
      std::cout << "size line error in MMF file.\n";
      exit(1);
    }
  }
  ParseElement(arguments, nr_rows_, nr_cols_, nr_nzeros_);
}

template <typename IndexType, typename ValueType>
void MMF<IndexType, ValueType>::DoLoadMmfMatrix() {
  Elem<IndexType, ValueType> elem;
  IndexType tmp;

  if (symmetric_) {
    matrix_.reserve(nr_nzeros_ << 1);
    for (IndexType i = 0; i < nr_nzeros_; i++) {
      if (!MMF::GetNext(elem.row, elem.col, elem.val)) {
        std::cout << "Requesting dereference, but mmf ended.\n";
        exit(1);
      }
      matrix_.push_back(elem);
      if (elem.row != elem.col) {
        tmp = elem.row;
        elem.row = elem.col;
        elem.col = tmp;
        matrix_.push_back(elem);
      }
    }
  } else {
    matrix_.reserve(nr_nzeros_);
    for (IndexType i = 0; i < nr_nzeros_; i++) {
      if (!MMF::GetNext(elem.row, elem.col, elem.val)) {
        std::cout << "Requesting dereference, but mmf ended.\n";
        exit(1);
      }
      matrix_.push_back(elem);
    }
  }

  sort(matrix_.begin(), matrix_.end(), ElemSorter<IndexType, ValueType>());
}

template <typename IndexType, typename ValueType>
bool MMF<IndexType, ValueType>::GetNext(IndexType &y, IndexType &x,
                                        ValueType &v) {
  std::vector<std::string> arguments;

  if (!DoRead(in, arguments)) {
    return false;
  }

  ParseElement(arguments, y, x, v);

  if (zero_based_) {
    y++;
    x++;
  }

  return true;
}

template <typename IndexType, typename ValueType>
void ParseElement(std::vector<std::string> &arguments, IndexType &y,
                  IndexType &x, ValueType &v) {
  if (arguments.size() >= 3) {
    y = atoi(arguments[0].c_str());
    x = atoi(arguments[1].c_str());
    v = atof(arguments[2].c_str());
  } else if (arguments.size() == 2) {
    y = atoi(arguments[0].c_str());
    x = atoi(arguments[1].c_str());
    v = 0.42;
  } else {
    std::cout << arguments[0] << std::endl;
    std::cout << "bad input, less arguments in line of MMF file.\n";
    exit(1);
  }
}

} // end of namespace io
} // end of namespace util
