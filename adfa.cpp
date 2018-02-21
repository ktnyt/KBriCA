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
      : b(n_output), f(f), lr(lr) {
    W = MatrixXf::Random(n_input, n_output);
    float max = W.maxCoeff();
    W /= (max * sqrt(static_cast<float>(n_input)));
  }

  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer input = inputs[0];
    Buffer error = inputs[1];

    if (error.len()) {
      MatrixXf x = matrixFromBuffer(input_queue.front());
      MatrixXf y = matrixFromBuffer(output_queue.front());
      input_queue.pop();
      output_queue.pop();

      MatrixXf e = matrixFromBuffer(error);

      MatrixXf d_x = (e * B).array() * f.backward(y).array();
      MatrixXf d_W = -x.transpose() * d_x;

      W += d_W * lr;

      for (int i = 0; i < d_x.cols(); ++i) {
        b(i) += d_x.col(i).sum() * lr;
      }
    }

    if (input.len()) {
      MatrixXf x = matrixFromBuffer(input);
      MatrixXf a = x * W;
      a.transpose().colwise() += b;
      MatrixXf y = f.forward(y);
      Buffer output = bufferFromMatrix(y);

      input_queue.push(input);
      output_queue.push(output);

      return output;
    }

    return Buffer();
  }

 private:
  MatrixXf W;
  VectorXf b;

  MatrixXf B;

  std::queue<Buffer> input_queue;
  std::queue<Buffer> output_queue;

  Function& f;

  float lr;
};

class Subtract : public Functor {
 public:
  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer a = inputs[0];
    Buffer b = inputs[1];

    if (a.len()) {
      a_queue.push(a);
    }

    if (b.len()) {
      b_queue.push(b);
    }

    if (a_queue.size() && b_queue.size()) {
      MatrixXf x = matrixFromBuffer(a_queue.front());
      MatrixXf y = matrixFromBuffer(b_queue.front());
      MatrixXf z = x - y;
      return bufferFromMatrix(z);
    }

    return Buffer();
  }

 private:
  std::queue<Buffer> a_queue;
  std::queue<Buffer> b_queue;
};

class CrossEntropy : public Functor {
 public:
  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer input = inputs[0];
    Buffer label = inputs[1];

    if (input.len()) {
      input_queue.push(input);
    }

    if (label.len()) {
      label_queue.push(label);
    }

    if (input_queue.size() && label_queue.size()) {
      MatrixXf y = matrixFromBuffer(input_queue.front());
      MatrixXf t = matrixFromBuffer(label_queue.front());
      input_queue.pop();
      label_queue.pop();

      Buffer buffer(sizeof(float));
      *reinterpret_cast<float*>(buffer.get()) = cross_entropy(y, t);

      return buffer;
    }

    return Buffer();
  }

 private:
  std::queue<Buffer> input_queue;
  std::queue<Buffer> label_queue;
};

class Accuracy : public Functor {
 public:
  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer input = inputs[0];
    Buffer label = inputs[1];

    if (input.len()) {
      input_queue.push(input);
    }

    if (label.len()) {
      label_queue.push(label);
    }

    if (input_queue.size() && label_queue.size()) {
      MatrixXf y = matrixFromBuffer(input_queue.front());
      MatrixXf t = matrixFromBuffer(label_queue.front());
      input_queue.pop();
      label_queue.pop();

      Buffer buffer(sizeof(float));
      *reinterpret_cast<float*>(buffer.get()) = accuracy(y, t);

      return buffer;
    }

    return Buffer();
  }

 private:
  std::queue<Buffer> input_queue;
  std::queue<Buffer> label_queue;
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
  Subtract subtract_f;
  Accuracy accuracy_f;
  CrossEntropy cross_entropy_f;

  Component* images_c = new Component(images_f, 0);
  Component* labels_c = new Component(labels_f, 0);
  Component* layer_c = new Component(layer_f, 1);
  Component* subtract_c = new Component(subtract_f, 2);
  Component* accuracy_c = new Component(accuracy_f, 2);
  Component* cross_entropy_c = new Component(cross_entropy_f, 2);

  layer_c->addConnection(images_c);
  layer_c->addConnection(subtract_c);

  subtract_c->addConnection(layer_c);
  subtract_c->addConnection(labels_c);

  accuracy_c->addConnection(layer_c);
  accuracy_c->addConnection(labels_c);

  cross_entropy_c->addConnection(layer_c);
  cross_entropy_c->addConnection(labels_c);

  VTSScheduler s;

  s.addComponent(images_c);
  s.addComponent(labels_c);
  s.addComponent(layer_c);
  s.addComponent(subtract_c);
  s.addComponent(accuracy_c);
  s.addComponent(cross_entropy_c);

  s.step();
  s.step();
  s.step();
  s.step();

  MPI_Finalize();

  return 0;
}
