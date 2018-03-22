#ifndef __ADFA_UTILS_HPP__
#define __ADFA_UTILS_HPP__

#include <algorithm>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "Eigen/Core"

Eigen::VectorXf softmax(Eigen::VectorXf v) {
  Eigen::VectorXf max = Eigen::VectorXf::Ones(v.size()) * v.maxCoeff();
  Eigen::VectorXf r = (v - max).unaryExpr(&expf);
  return r / r.sum();
}

float accuracy(Eigen::MatrixXf y, Eigen::MatrixXf t) {
  float total = 0.0f;
  for (std::size_t i = 0; i < y.rows(); ++i) {
    Eigen::MatrixXf::Index max_y, max_t, tmp;
    y.row(i).maxCoeff(&tmp, &max_y);
    t.row(i).maxCoeff(&tmp, &max_t);
    if (max_y == max_t) {
      total += 1.0;
    }
  }
  return total / static_cast<float>(y.rows());
}

float cross_entropy(Eigen::MatrixXf y, Eigen::MatrixXf t) {
  return -(y.array() * ((t.array() + 1e-10f).log())).sum() / t.rows();
}

float mean_squared_error(Eigen::MatrixXf y, Eigen::MatrixXf t) {
  Eigen::MatrixXf d = y - t;
  Eigen::VectorXf f(Eigen::Map<Eigen::VectorXf>(d.data(), d.cols() * d.rows()));
  return f.dot(f) / d.size();
}

template <typename Vector>
void serialize_vector(const char* filename, Vector& v) {
  typename Vector::Index size = v.size();

  size_t index_size = sizeof(typename Vector::Index);
  size_t value_size = sizeof(typename Vector::Scalar);

  size_t body_size = value_size * size;
  size_t total_size = index_size + body_size;

  char* buffer = new char[total_size];
  char* psize = reinterpret_cast<char*>(&size);
  char* data = reinterpret_cast<char*>(v.data());

  std::copy(psize, psize + index_size, buffer);
  std::copy(data, data + body_size, buffer + index_size);

  std::ofstream f(filename, std::ios::out | std::ios::binary);

  if (!f.is_open()) {
    std::ostringstream os;
    os << "Failed to open file: " << filename;
    throw std::runtime_error(os.str());
  }

  f.write(buffer, total_size);

  f.close();
}

template <typename Vector>
void deserialize_vector(const char* filename, Vector& v) {
  std::ifstream f(filename, std::ios::in | std::ios::binary | std::ios::ate);

  if (!f.is_open()) {
    std::ostringstream os;
    os << "Failed to open file: " << filename;
    throw std::runtime_error(os.str());
  }

  f.seekg(0, std::ios::beg);

  typename Vector::Index size;

  size_t index_size = sizeof(typename Vector::Index);
  size_t value_size = sizeof(typename Vector::Scalar);

  char* psize = reinterpret_cast<char*>(&size);

  f.read(psize, index_size);

  v.resize(size);

  char* data = reinterpret_cast<char*>(v.data());

  f.read(data, value_size * size);
  f.close();
}

template <typename Matrix>
void serialize_matrix(const char* filename, Matrix& m) {
  typename Matrix::Index rows = m.rows();
  typename Matrix::Index cols = m.cols();

  size_t index_size = sizeof(typename Matrix::Index);
  size_t value_size = sizeof(typename Matrix::Scalar);

  size_t header_size = index_size * 2;
  size_t body_size = value_size * rows * cols;
  size_t total_size = header_size + body_size;

  char* buffer = new char[total_size];
  char* prows = reinterpret_cast<char*>(&rows);
  char* pcols = reinterpret_cast<char*>(&cols);
  char* data = reinterpret_cast<char*>(m.data());

  std::copy(prows, prows + index_size, buffer);
  std::copy(pcols, pcols + index_size, buffer + index_size);
  std::copy(data, data + body_size, buffer + header_size);

  std::ofstream f(filename, std::ios::out | std::ios::binary);

  if (!f.is_open()) {
    std::ostringstream os;
    os << "Failed to open file: " << filename;
    throw std::runtime_error(os.str());
  }

  f.write(buffer, total_size);

  f.close();
}

template <typename Matrix>
void deserialize_matrix(const char* filename, Matrix& m) {
  std::ifstream f(filename, std::ios::in | std::ios::binary | std::ios::ate);

  if (!f.is_open()) {
    std::ostringstream os;
    os << "Failed to open file: " << filename;
    throw std::runtime_error(os.str());
  }

  f.seekg(0, std::ios::beg);

  typename Matrix::Index rows;
  typename Matrix::Index cols;

  size_t index_size = sizeof(typename Matrix::Index);
  size_t value_size = sizeof(typename Matrix::Scalar);

  char* prows = reinterpret_cast<char*>(&rows);
  char* pcols = reinterpret_cast<char*>(&cols);

  f.read(prows, index_size);
  f.read(pcols, index_size);

  m.resize(rows, cols);

  char* data = reinterpret_cast<char*>(m.data());

  f.read(data, value_size * rows * cols);
  f.close();
}

#endif
