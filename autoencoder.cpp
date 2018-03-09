#include <Eigen/Core>
#include <cmath>
#include <iostream>

#include "activations.hpp"
#include "mnist.hpp"
#include "random.hpp"
#include "utils.hpp"

using namespace Eigen;

MatrixXf linear(MatrixXf& x, MatrixXf& W, VectorXf& b) {
  MatrixXf a = x * W;
  a.transpose().colwise() += b;
  return a;
}

class Autoencoder {
 public:
  Autoencoder(int n_input, int n_hidden, float lr = 0.05)
      : W(n_input, n_hidden),
        U(n_hidden, n_input),
        b(VectorXf::Zero(n_hidden)),
        c(VectorXf::Zero(n_input)),
        lr(lr) {
    float stdW = 1. / sqrt(static_cast<float>(n_input));
    float stdU = 1. / sqrt(static_cast<float>(n_hidden));

    Normal<float> genW(0, stdW);
    Normal<float> genU(0, stdU);

    for (int i = 0; i < n_input; ++i) {
      for (int j = 0; j < n_hidden; ++j) {
        W(i, j) = genW();
        U(j, i) = genU();
      }
    }
  }

  float operator()(MatrixXf x) {
    MatrixXf y = linear(x, W, b).unaryExpr(&sigmoid);
    MatrixXf z = linear(y, U, c).unaryExpr(&sigmoid);

    MatrixXf d_z = z - x;
    MatrixXf d_y = (d_z * U.transpose()).array() * y.unaryExpr(&dsigmoid).array();

    MatrixXf d_W = -x.transpose() * d_y;
    MatrixXf d_U = -y.transpose() * d_z;

    W += d_W * lr;
    U += d_U * lr;

    for (int i = 0; i < d_y.cols(); ++i) {
      // b(i) += d_y.col(i).sum() * lr;
    }

    for (int i = 0; i < d_z.cols(); ++i) {
      // c(i) += d_z.col(i).sum() * lr;
    }

    return mean_squared_error(z, x);
  }

 private:
  MatrixXf W;
  MatrixXf U;
  VectorXf b;
  VectorXf c;

  float lr;
};

int main() {
  MNIST<float> mnist("./mnist");
  mnist.train_images.scale(256.0, true);

  int length = mnist.train_images.length;
  int* perm = new int[length];

  for (int i = 0; i < length; ++i) {
    perm[i] = i;
  }

  Autoencoder ae(mnist.train_images.dims, 1000);

  int batchsize = 100;

  for (int epoch = 0; epoch < 20; ++epoch) {
    shuffle(perm, perm + length);
    mnist.reorder(perm);

    float loss = 0.0;
    for (int batch = 0; batch < length; batch += batchsize) {
      if ((batch + batchsize) % 800 == 0) std::cerr << "#";
      float* images = mnist.train_images.getBatch(batch, batchsize);
      Map<MatrixXf> x(images, batchsize, mnist.train_images.dims);
      loss += ae(x) * batchsize;
    }
    std::cerr << std::endl;
    std::cout << epoch << " " << loss / length << std::endl;
  }

  return 0;
}
