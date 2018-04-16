#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>

#include "kbrica.hpp"

#include "Eigen/Core"
#include "activations.hpp"
#include "mnist.hpp"
#include "mt64.hpp"
#include "random.hpp"
#include "utils.hpp"

using namespace kbrica;
using namespace Eigen;

void print_mnist(float* image, float* label) {
  for (std::size_t i = 0; i < 28; ++i) {
    for (std::size_t j = 0; j < 28; ++j) {
      float pixel = image[i * 28 + j];
      if (pixel < 0.25) {
        std::cout << "  ";
      } else if (pixel < 0.5) {
        std::cout << "**";
      } else if (pixel < 0.75) {
        std::cout << "OO";
      } else {
        std::cout << "@@";
      }
    }
    std::cout << std::endl;
  }

  for (std::size_t i = 0; i < 10; ++i) {
    std::cout << i << " " << static_cast<int>(label[i]) << std::endl;
  }
}

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

MatrixXf uniform(std::size_t rows, std::size_t cols) {
  float scale = 1. / static_cast<float>(rows);
  MatrixXf mat = Eigen::MatrixXf::Random(rows, cols);
  mat *= scale / mat.maxCoeff();
  return mat;
}

class Layer {
 public:
  virtual MatrixXf forward(MatrixXf) = 0;
  virtual void backward(MatrixXf) = 0;
};

class LayerFunctor : public Functor {
 public:
  LayerFunctor(Layer* layer) : layer(layer) {}

  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer buffer;

    if (inputs[0].size()) {
      MatrixXf x = matrixFromBuffer(inputs[0]);
      MatrixXf y = layer->forward(x);
      buffer = bufferFromMatrix(y);
    }

    if (inputs[1].size()) {
      MatrixXf e = matrixFromBuffer(inputs[1]);
      layer->backward(e);
    }

    return buffer;
  }

 private:
  Layer* layer;
};

class ErrorFunctor : public Functor {
 public:
  ErrorFunctor() : loss(0.0), acc(0.0), count(0) {}

  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer buffer;

    if (inputs[0].size()) {
      MatrixXf y = matrixFromBuffer(inputs[0]);
      ys.push(y);
    }

    if (inputs[1].size()) {
      MatrixXf t = matrixFromBuffer(inputs[1]);
      ts.push(t);
    }

    if (ys.size() && ts.size()) {
      MatrixXf y = ys.front();
      MatrixXf t = ts.front();
      ys.pop();
      ts.pop();

      MatrixXf e = y - t;
      loss += cross_entropy(y, t);
      acc += accuracy(y, t);

      buffer = bufferFromMatrix(e);
    }

    return buffer;
  }

 private:
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

class HiddenLayer : public Layer {
 public:
  HiddenLayer(std::size_t n_input, std::size_t n_output, std::size_t n_final)
      : W(uniform(n_input, n_output)), B(uniform(n_final, n_output)) {}

  MatrixXf forward(MatrixXf x) {
    MatrixXf y = (x * W).unaryExpr(&sigmoid);
    xs.push(x);
    ys.push(y);
    return y;
  }

  void backward(MatrixXf e) {
    MatrixXf x = xs.front();
    MatrixXf y = ys.front();
    xs.pop();
    ys.pop();

    MatrixXf d_y = (e * B).array() * y.unaryExpr(&dsigmoid).array();
    MatrixXf d_W = -x.transpose() * d_y;

    W += d_W * 0.05;
  }

 private:
  MatrixXf W;
  MatrixXf B;
  std::queue<MatrixXf> xs;
  std::queue<MatrixXf> ys;
};

class OutputLayer : public Layer {
 public:
  OutputLayer(std::size_t n_input, std::size_t n_output)
      : W(uniform(n_input, n_output)) {}

  MatrixXf forward(MatrixXf x) {
    xs.push(x);
    return (x * W).unaryExpr(&sigmoid);
  }

  void backward(MatrixXf e) {
    MatrixXf x = xs.front();
    xs.pop();

    MatrixXf d_W = -x.transpose() * e;
    W += d_W * 0.05;
  }

 private:
  MatrixXf W;
  std::queue<MatrixXf> xs;
};

int main() {
  std::vector<std::vector<unsigned char> > train_images;
  std::vector<unsigned char> train_labels;
  train_images = read_image("mnist/train-images-idx3-ubyte");
  train_labels = read_label("mnist/train-labels-idx1-ubyte");

  std::vector<std::vector<unsigned char> > test_images;
  std::vector<unsigned char> test_labels;
  test_images = read_image("mnist/t10k-images-idx3-ubyte");
  test_labels = read_label("mnist/t10k-labels-idx1-ubyte");

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

  std::size_t* perm = new std::size_t[N_train];
  for (std::size_t i = 0; i < N_train; ++i) {
    perm[i] = i;
  }

  std::size_t n_epoch = 20;
  std::size_t batchsize = 100;
  std::size_t n_hidden = 480;
  std::size_t n_output = 10;

  HiddenLayer l1(x_shape, n_hidden, n_output);
  HiddenLayer l2(n_hidden, n_hidden, n_output);
  HiddenLayer l3(n_hidden, n_hidden, n_output);
  OutputLayer l4(n_hidden, n_output);

  LayerFunctor f1(&l1);
  LayerFunctor f2(&l2);
  LayerFunctor f3(&l3);
  LayerFunctor f4(&l4);
  ErrorFunctor f5;
  Pipe pipe;

  Component c1(f1, 0);
  Component c2(f2, 1);
  Component c3(f3, 2);
  Component c4(f4, 3);
  Component c5(f5, 3);
  Component ci(pipe, 0);
  Component cl(pipe, 0);

  c1.connect(&ci);
  c1.connect(&c5);

  c2.connect(&c1);
  c2.connect(&c5);

  c3.connect(&c2);
  c3.connect(&c5);

  c4.connect(&c3);
  c4.connect(&c5);

  c5.connect(&c4);
  c5.connect(&cl);

  MT19937 rng;

  for (std::size_t epoch = 0; epoch < n_epoch; ++epoch) {
    float loss = 0.0;
    float acc = 0.0;
    std::size_t count = 0;

    shuffle(perm, perm + N_train, rng);
    for (std::size_t batchnum = 0; batchnum < N_train; batchnum += batchsize) {
      if ((batchnum + batchsize) % 800 == 0) std::cerr << "#";

      MatrixXf x(batchsize, x_shape);
      MatrixXf t(batchsize, y_shape);

      for (std::size_t i = 0; i < batchsize; ++i) {
        x.row(i) = x_train.row(perm[batchnum + i]);
        t.row(i) = y_train.row(perm[batchnum + i]);
      }

      MatrixXf h1 = l1.forward(x);
      MatrixXf h2 = l2.forward(h1);
      MatrixXf h3 = l3.forward(h2);
      MatrixXf y = l4.forward(h3);

      loss += cross_entropy(t, y);
      acc += accuracy(t, y);

      MatrixXf e = y - t;
      l1.backward(e);
      l2.backward(e);
      l3.backward(e);
      l4.backward(e);

      ++count;
    }

    std::cerr << std::endl;

    std::cout << "Loss: " << std::setprecision(7) << loss / count
              << " Accuracy: " << acc / count << std::endl;
  }

  return 0;
}
