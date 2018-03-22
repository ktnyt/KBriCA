#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "kbrica.hpp"
#include "mnist/mnist_reader.hpp"
#include "mpi.h"

#include "activations.hpp"
#include "functions.hpp"
#include "utils.hpp"

using namespace Eigen;
using namespace kbrica;

Buffer bufferFromMatrix(MatrixXf& m) {
  std::size_t head_size = 2 * sizeof(std::size_t);
  std::size_t data_size = m.size() * sizeof(float);
  Buffer buffer(head_size + data_size);
  char* data = buffer.data();
  *reinterpret_cast<std::size_t*>(data) = m.rows();
  data += sizeof(std::size_t);
  *reinterpret_cast<std::size_t*>(data) = m.cols();
  data += sizeof(std::size_t);
  float* casted = reinterpret_cast<float*>(data);
  std::copy(m.data(), m.data() + m.size(), casted);
  return buffer;
}

MatrixXf matrixFromBuffer(Buffer& buffer) {
  char* data = buffer.data();
  std::size_t rows = *reinterpret_cast<std::size_t*>(data);
  data += sizeof(std::size_t);
  std::size_t cols = *reinterpret_cast<std::size_t*>(data);
  MatrixXf m(rows, cols);
  float* casted = reinterpret_cast<float*>(data);
  std::copy(casted, casted + m.size(), m.data());
  return m;
}

static std::random_device seed_gen;

MatrixXf linear(MatrixXf& x, MatrixXf& W, VectorXf& b) {
  MatrixXf a = x * W;
  a.transpose().colwise() += b;
  return a;
}

class DFALayer : public Functor {
  static constexpr float epsilon = std::numeric_limits<float>::epsilon();

 public:
  DFALayer(std::size_t n_input, std::size_t n_output, std::size_t n_final,
           float lr, float ae, float decay)
      : W(n_input, n_output),
        U(n_output, n_input),
        B(n_final, n_output),
        b(VectorXf::Zero(n_output)),
        c(VectorXf::Zero(n_input)),
        lr(lr),
        ae(ae),
        decay(decay),
        loss(0.0),
        count(0) {
    float stdW = 1. / sqrt(static_cast<float>(n_input));
    float stdB = 1. / sqrt(static_cast<float>(n_output));

    std::default_random_engine engine(seed_gen());

    std::uniform_real_distribution<float> genW(-stdW, stdW);
    std::uniform_real_distribution<float> genB(-stdB, stdB);

    for (int j = 0; j < n_output; ++j) {
      for (int i = 0; i < n_input; ++i) {
        W(i, j) = genW(engine);
      }
      for (int i = 0; i < n_final; ++i) {
        B(i, j) = genB(engine);
      }
    }
  }

  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer ret;

    if (inputs[0].size()) {
      MatrixXf x = matrixFromBuffer(inputs[0]);
      MatrixXf y = linear(x, W, b).unaryExpr(&sigmoid);

      xs.push(x);
      ys.push(y);

      ret = bufferFromMatrix(y);

      if (ae > epsilon) {
        MatrixXf U = W.transpose();
        MatrixXf z = linear(y, U, c).unaryExpr(&sigmoid);

        MatrixXf d_z = z - x;
        MatrixXf d_y = (d_z * W).array() * y.unaryExpr(&dsigmoid).array();

        MatrixXf d_W = -(x.transpose() * d_y + d_z.transpose() * y);

        // W += d_W * ae;

        for (int i = 0; i < d_y.cols(); ++i) {
          // b(i) += d_y.col(i).sum() * ae;
        }

        for (int i = 0; i < d_z.cols(); ++i) {
          // c(i) += d_z.col(i).sum() * ae;
        }

        ae *= decay;

        loss += mean_squared_error(z, x);
        ++count;
      }
    }

    if (inputs[1].size()) {
      MatrixXf x = xs.front();
      MatrixXf y = ys.front();
      xs.pop();
      ys.pop();

      MatrixXf e = matrixFromBuffer(inputs[1]);
      MatrixXf d_y = e * B;
      MatrixXf d_x = d_y.array() * y.unaryExpr(&dsigmoid).array();
      MatrixXf d_W = -x.transpose() * d_x;
      W += d_W * lr;
      for (int i = 0; i < d_x.cols(); ++i) {
        // b(i) += d_x.col(i).sum() * lr;
      }
    }

    return ret;
  }

 private:
  MatrixXf W;
  MatrixXf U;
  MatrixXf B;
  VectorXf b;
  VectorXf c;
  float lr;
  float ae;
  float decay;
  std::queue<MatrixXf> xs;
  std::queue<MatrixXf> ys;

 public:
  float loss;
  std::size_t count;
};

class OutLayer : public Functor {
 public:
  OutLayer(std::size_t n_input, std::size_t n_output, float lr)
      : W(n_input, n_output),
        b(VectorXf::Zero(n_output)),
        lr(lr),
        loss(0.0),
        acc(0.0),
        count(0) {
    float stdW = 1. / sqrt(static_cast<float>(n_input));

    std::default_random_engine engine(seed_gen());

    std::uniform_real_distribution<float> genW(-stdW, stdW);

    for (int i = 0; i < n_input; ++i) {
      for (int j = 0; j < n_output; ++j) {
        W(i, j) = genW(engine);
      }
    }
  }

  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer ret;

    if (inputs[0].size()) {
      MatrixXf x = matrixFromBuffer(inputs[0]);
      MatrixXf y = linear(x, W, b).unaryExpr(&sigmoid);
      // for (std::size_t i = 0; i < y.rows(); ++i) {
      //   y.row(i) = softmax(y.row(i));
      // }

      xs.push(x);
      ys.push(y);
    }

    if (inputs[1].size()) {
      MatrixXf t = matrixFromBuffer(inputs[1]);
      ts.push(t);
    }

    if (xs.size() && ys.size() && ts.size()) {
      MatrixXf x = xs.front();
      MatrixXf y = ys.front();
      MatrixXf t = ts.front();
      xs.pop();
      ys.pop();
      ts.pop();

      loss += cross_entropy(t, y);
      acc += accuracy(t, y);
      ++count;

      MatrixXf d_y = y - t;
      MatrixXf d_W = -x.transpose() * d_y;
      W += d_W * lr;

      for (int i = 0; i < d_y.cols(); ++i) {
        // b(i) += d_y.col(i).sum() * lr;
      }

      ret = bufferFromMatrix(d_y);
    }

    return ret;
  }

 private:
  MatrixXf W;
  VectorXf b;
  float lr;
  std::queue<MatrixXf> xs;
  std::queue<MatrixXf> ys;
  std::queue<MatrixXf> ts;

 public:
  float loss;
  float acc;
  std::size_t count;
};

class Pipe : public Functor {
 public:
  Buffer operator()(std::vector<Buffer>& inputs) { return inputs[0]; }
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

  MatrixXf x_train(N_train, x_shape);
  MatrixXf y_train(N_train, y_shape);

  MatrixXf x_test(N_test, x_shape);
  MatrixXf y_test(N_test, y_shape);

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

  std::size_t n_epoch = 20;
  std::size_t batchsize = 100;
  std::size_t n_hidden = 480;

  Pipe pipe;

  DFALayer layer0(x_shape, n_hidden, 10, 0.01, 0.01, 0.9995);
  DFALayer layer1(n_hidden, n_hidden, 10, 0.01, 0.01, 0.9995);
  DFALayer layer2(n_hidden, n_hidden, 10, 0.01, 0.01, 0.9995);
  OutLayer layer3(n_hidden, 10, 0.01);

  Component image_pipe(pipe, 0);
  Component label_pipe(pipe, 0);
  Component component0(layer0, 0);
  Component component1(layer1, 1);
  Component component2(layer2, 2);
  Component component3(layer3, 3);

  component0.connect(&image_pipe, &component3);
  // component1.connect(&component0, &component3);
  // component2.connect(&component1, &component3);
  component3.connect(&component0, &label_pipe);

  VTSScheduler s;
  s.addComponent(&image_pipe);
  s.addComponent(&label_pipe);
  s.addComponent(&component0);
  // s.addComponent(&component1);
  // s.addComponent(&component2);
  s.addComponent(&component3);

  for (std::size_t epoch = 0; epoch < n_epoch; ++epoch) {
    if (rank == 0) {
      std::random_shuffle(perm.begin(), perm.end());
    }

    for (std::size_t batchnum = 0; batchnum < N_train; batchnum += batchsize) {
      if (rank == 0) {
        if ((batchnum + batchsize) % 800 == 0) std::cerr << "#";
      }

      std::vector<std::size_t> indices(perm.begin() + batchnum,
                                       perm.begin() + batchnum + batchsize);

      MatrixXf x(batchsize, x_shape);
      MatrixXf t(batchsize, y_shape);

      for (std::size_t i = 0; i < batchsize; ++i) {
        x.row(i) = x_train.row(indices[i]);
        t.row(i) = y_train.row(indices[i]);
      }

      image_pipe.setInput(0, bufferFromMatrix(x));
      label_pipe.setInput(0, bufferFromMatrix(t));

      s.step();
    }

    if (rank == 0) {
      std::cerr << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "Train\t L0 Loss: " << std::setprecision(7)
                << layer0.loss / layer0.count << std::endl;
      layer0.loss = 0.0;
      layer0.count = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
      std::cout << "Train\t L1 Loss: " << std::setprecision(7)
                << layer1.loss / layer1.count << std::endl;
      layer1.loss = 0.0;
      layer1.count = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 2) {
      std::cout << "Train\t L2 Loss: " << std::setprecision(7)
                << layer2.loss / layer2.count << std::endl;
      layer2.loss = 0.0;
      layer2.count = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 3) {
      std::cout << "Train\t L3 Loss: " << std::setprecision(7)
                << layer3.loss / layer3.count
                << " Accuracy: " << layer3.acc / layer3.count << std::endl;
      layer3.loss = 0.0;
      layer3.acc = 0.0;
      layer3.count = 0;
    }
  }

  MPI_Finalize();

  return 0;
}
