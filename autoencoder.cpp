#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "mnist/mnist_reader.hpp"

#include "activations.hpp"
#include "utils.hpp"

using namespace Eigen;

static std::random_device seed_gen;

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

    std::default_random_engine engine(seed_gen());

    std::normal_distribution<float> genW(0.0, stdW);
    std::normal_distribution<float> genU(0.0, stdU);

    for (int i = 0; i < n_input; ++i) {
      for (int j = 0; j < n_hidden; ++j) {
        W(i, j) = genW(engine);
        U(j, i) = genU(engine);
      }
    }
  }

  float operator()(MatrixXf x) {
    MatrixXf y = linear(x, W, b).unaryExpr(&sigmoid);
    MatrixXf z = linear(y, U, c).unaryExpr(&sigmoid);

    MatrixXf d_z = z - x;
    MatrixXf d_y =
        (d_z * U.transpose()).array() * y.unaryExpr(&dsigmoid).array();

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
  auto mnist = mnist::read_dataset<std::vector, std::vector, unsigned char,
                                   unsigned char>();

  std::vector<std::vector<unsigned char> > train_images = mnist.training_images;
  std::vector<unsigned char> train_labels = mnist.training_labels;

  std::vector<std::vector<unsigned char> > test_images = mnist.test_images;
  std::vector<unsigned char> test_labels = mnist.test_labels;

  std::size_t N_train = train_images.size();
  std::size_t N_test = test_images.size();
  std::size_t x_shape = train_images[0].size();
  std::size_t y_shape = 10;
  std::size_t n_epoch = 200;

  Eigen::MatrixXf x_train(N_train, x_shape);
  Eigen::MatrixXf y_train(N_train, y_shape);

  Eigen::MatrixXf x_test(N_test, x_shape);
  Eigen::MatrixXf y_test(N_test, y_shape);

  for (std::size_t i = 0; i < N_train; ++i) {
    for (std::size_t j = 0; j < x_shape; ++j) {
      x_train(i, j) = static_cast<float>(train_images[i][j]) / 255.0;
    }
    for (std::size_t j = 0; j < y_shape; ++j) {
      y_train(i, j) = train_labels[i] == j ? 1.0f : 0.0f;
    }
    if (i < N_test) {
      for (std::size_t j = 0; j < x_shape; ++j) {
        x_test(i, j) = static_cast<float>(test_images[i][j]) / 255.0;
      }
      for (std::size_t j = 0; j < y_shape; ++j) {
        y_test(i, j) = test_labels[i] == j ? 1.0f : 0.0f;
      }
    }
  }

  std::vector<std::size_t> perm;
  for (std::size_t i = 0; i < N_train; ++i) {
    perm.push_back(i);
  }

  std::size_t batchsize = 100;

  Autoencoder ae(784, 1000);

  for (std::size_t epoch = 0; epoch < n_epoch; ++epoch) {
    std::cout << "Epoch: " << epoch + 1 << std::endl;
    std::random_shuffle(perm.begin(), perm.end());

    float loss = 0.0f;

    for (std::size_t i = 0; i < N_train; i += batchsize) {
      if ((i + batchsize) % 800 == 0) std::cerr << "#";
      std::vector<std::size_t> indices(perm.begin() + i,
                                       perm.begin() + i + batchsize);

      Eigen::MatrixXf x(batchsize, x_shape);
      Eigen::MatrixXf t(batchsize, y_shape);

      for (std::size_t i = 0; i < batchsize; ++i) {
        x.row(i) = x_train.row(indices[i]);
        t.row(i) = y_train.row(indices[i]);
      }

      loss += ae(x) * batchsize;
    }

    std::cerr << std::endl;
    std::cout << "Train\tLoss: " << std::setprecision(7) << loss / perm.size()
              << std::endl;
  }

  return 0;
}
