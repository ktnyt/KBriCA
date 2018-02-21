#ifndef __ADFA_FUNCTIONS_HPP__
#define __ADFA_FUNCTIONS_HPP__

#include "Eigen/Core"

#include "activations.hpp"
#include "utils.hpp"

using namespace Eigen;

class Function {
 public:
  virtual MatrixXf forward(MatrixXf&) = 0;
  virtual MatrixXf backward(MatrixXf&) = 0;
};

struct Sigmoid : public Function {
  MatrixXf forward(MatrixXf& x) { return x.unaryExpr(&sigmoid); }
  MatrixXf backward(MatrixXf& y) { return y.unaryExpr(&dsigmoid); }
};

struct Softmax : public Function {
  MatrixXf forward(MatrixXf& x) {
    MatrixXf y(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
      y.row(i) = softmax(x.row(i));
    }
    return y;
  }

  MatrixXf backward(MatrixXf& y) { return MatrixXf::Ones(y.rows(), y.cols()); }
};

#endif
