#include <algorithm>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <vector>

#include "Eigen/Core"
#include "kbrica.hpp"
#include "mnist.hpp"
#include "mpi.h"

#include "activations.hpp"
#include "functions.hpp"
#include "utils.hpp"

using namespace Eigen;
using namespace kbrica;

template <typename MatrixT>
Buffer bufferFromMatrix(MatrixT& m) {
  Buffer buffer(sizeof(int) * 2 + sizeof(float) * m.size());

  char* data = buffer.get();

  *reinterpret_cast<int*>(data) = m.rows();
  data += sizeof(int);

  *reinterpret_cast<int*>(data) = m.cols();
  data += sizeof(int);

  float* casted = reinterpret_cast<float*>(data);

  std::copy(m.data(), m.data() + m.size(), casted);

  return buffer;
}

MatrixXf matrixFromBuffer(Buffer& b) {
  char* data = b.get();

  int rows = *reinterpret_cast<int*>(data);
  data += sizeof(int);

  int cols = *reinterpret_cast<int*>(data);
  data += sizeof(int);

  Map<MatrixXf> map(reinterpret_cast<float*>(data), rows, cols);

  MatrixXf matrix(rows, cols);

  float* casted = reinterpret_cast<float*>(data);

  std::copy(casted, casted + rows * cols, matrix.data());

  return map;
}

class ImageLoader : public Functor {
 public:
  ImageLoader(MNIST<float>& mnist, int batchsize)
      : mnist(mnist), batchsize(batchsize) {}
  Buffer operator()(std::vector<Buffer>& inputs) {
    float* images = mnist.train_images.getBatch(batchnum, batchsize);

    Map<MatrixXf> x(images, batchsize, mnist.train_images.dims);

    batchnum += batchsize;

    if (batchnum >= batchsize) {
      batchnum = 0;
    }

    return bufferFromMatrix(x);
  }

 private:
  MNIST<float>& mnist;

  int batchsize;
  int batchnum;
};

class LabelLoader : public Functor {
 public:
  LabelLoader(MNIST<float>& mnist, int batchsize)
      : mnist(mnist), batchsize(batchsize) {}
  Buffer operator()(std::vector<Buffer>& inputs) {
    float* labels = mnist.train_labels.getBatch(batchnum, batchsize);

    Map<MatrixXf> y(labels, batchsize, 10);

    batchnum += batchsize;

    if (batchnum >= batchsize) {
      batchnum = 0;
    }

    return bufferFromMatrix(y);
  }

 private:
  MNIST<float>& mnist;

  int batchsize;
  int batchnum;
};

class Layer : public Functor {
 public:
  Layer(int n_input, int n_output, Function& f, float lr)
      : b(n_output), c(n_input), batch(0), loss(0), f(f), lr(lr) {
    {
      W = MatrixXf::Random(n_input, n_output);
      float max = W.maxCoeff();
      W /= (max * sqrt(static_cast<float>(n_input)));
    }
    {
      U = MatrixXf::Random(n_output, n_input);
      float max = U.maxCoeff();
      U /= (max * sqrt(static_cast<float>(n_output)));
    }
  }

  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer input = inputs[0];

    if (input.len()) {
      MatrixXf x = matrixFromBuffer(input);
      MatrixXf a = x * W;
      a.transpose().colwise() += b;
      MatrixXf y = a.unaryExpr(&sigmoid);
      Buffer output = bufferFromMatrix(y);

      MatrixXf t = y * U;
      t.transpose().colwise() += c;
      MatrixXf z = t.unaryExpr(&sigmoid);

      MatrixXf d_z = z - x;
      MatrixXf d_y =
          (d_z * U.transpose()).array() * y.unaryExpr(&dsigmoid).array();

      MatrixXf d_W = -x.transpose() * d_y;
      MatrixXf d_U = -y.transpose() * d_z;

      W += d_W * lr;
      U += d_U * lr;

      for (int i = 0; i < d_y.cols(); ++i) {
        b(i) += d_y.col(i).sum() * lr;
      }

      for (int i = 0; i < d_z.cols(); ++i) {
        c(i) += d_z.col(i).sum() * lr;
      }

      loss += mean_squared_error(z, x) / static_cast<float>(x.rows());

      if (++batch == 600) {
        std::cout << loss << std::endl;
        loss = 0;
        batch = 0;
      }

      // input_queue.push(input);
      // output_queue.push(output);

      return output;
    }

    return Buffer();
  }

 private:
  MatrixXf W;
  VectorXf b;

  MatrixXf U;
  VectorXf c;

  MatrixXf B;

  int batch;
  float loss;

  // std::queue<Buffer> input_queue;
  // std::queue<Buffer> output_queue;

  Function& f;

  float lr;
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MNIST<float> mnist("./mnist");
  mnist.train_images.scale(256.0, true);

  int batchsize = 100;

  ImageLoader images_f(mnist, batchsize);
  LabelLoader labels_f(mnist, batchsize);

  Sigmoid sigmoid;

  Layer layer_f(mnist.train_images.dims, 10, sigmoid, 0.05);

  Component* images_c = new Component(images_f, 0);
  Component* labels_c = new Component(labels_f, 0);
  Component* layer_c = new Component(layer_f, 1);

  layer_c->addConnection(images_c);

  VTSScheduler s;

  s.addComponent(images_c);
  s.addComponent(labels_c);
  s.addComponent(layer_c);

  s.step();

  for (int i = 0; i < 600 * 10; ++i) {
    s.step();
  }

  MPI_Finalize();

  return 0;
}
