#include <iomanip>
#include <iostream>
#include <limits>
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

void print_mnist(VectorXf image) {
  for (std::size_t i = 0; i < 28; ++i) {
    for (std::size_t j = 0; j < 28; ++j) {
      float pixel = image(i * 28 + j);
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
}

class Timer {
 public:
  Timer() { reset(); }

  void reset() { clock_gettime(CLOCK_MONOTONIC, &ref); }

  int elapsed() {
    clock_gettime(CLOCK_MONOTONIC, &now);
    int sec = now.tv_sec - ref.tv_sec;
    int nsec = now.tv_nsec - ref.tv_nsec;
    return (sec * 1000 * 1000) + (nsec / 1000);
  }

 private:
  struct timespec ref;
  struct timespec now;
};

Buffer bufferFromMatrix(MatrixXf m) {
  std::size_t head_size = 2 * sizeof(std::size_t);
  std::size_t data_size = m.size() * sizeof(float);
  Buffer buffer(head_size + data_size);
  char* data = buffer.data();
  *reinterpret_cast<std::size_t*>(data) = m.rows();
  data += sizeof(std::size_t);
  *reinterpret_cast<std::size_t*>(data) = m.cols();
  data += sizeof(std::size_t);
  float* casted = reinterpret_cast<float*>(data);
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) {
      casted[i * m.cols() + j] = m(i, j);
    }
  }
  return buffer;
}

MatrixXf matrixFromBuffer(Buffer buffer) {
  char* data = buffer.data();
  std::size_t rows = *reinterpret_cast<std::size_t*>(data);
  data += sizeof(std::size_t);
  std::size_t cols = *reinterpret_cast<std::size_t*>(data);
  data += sizeof(std::size_t);
  MatrixXf m(rows, cols);
  float* casted = reinterpret_cast<float*>(data);
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) {
      m(i, j) = casted[i * m.cols() + j];
    }
  }
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
      ++count;

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
      : W(uniform(n_input, n_output)),
        B(uniform(n_final, n_output)),
        lr(0.01),
        epsilon(std::numeric_limits<float>::epsilon()),
        loss(0.0),
        count(0) {}

  MatrixXf forward(MatrixXf x) {
    MatrixXf y = (x * W).unaryExpr(&sigmoid);
    if (lr > epsilon) {
      MatrixXf z = (y * W.transpose()).unaryExpr(&sigmoid);
      MatrixXf d_z = z - x;
      loss += mean_squared_error(z, x);
      ++count;
      MatrixXf d_y = (d_z * W).array() * y.unaryExpr(&dsigmoid).array();
      MatrixXf d_W = -(x.transpose() * d_y + d_z.transpose() * y);
      W += d_W * lr;
      lr *= 0.9995;
    }
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
  float lr;
  float epsilon;

 public:
  float loss;
  std::size_t count;
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

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  std::vector<std::vector<unsigned char> > train_images;
  std::vector<unsigned char> train_labels;
  train_images = read_images("mnist/train-images-idx3-ubyte");
  train_labels = read_labels("mnist/train-labels-idx1-ubyte");

  std::vector<std::vector<unsigned char> > test_images;
  std::vector<unsigned char> test_labels;
  test_images = read_images("mnist/t10k-images-idx3-ubyte");
  test_labels = read_labels("mnist/t10k-labels-idx1-ubyte");

  std::size_t N_train = train_images.size();
  std::size_t N_test = test_images.size();
  std::size_t x_shape = train_images[0].size();
  std::size_t y_shape = 10;

  MatrixXf x_train(N_train, x_shape);
  MatrixXf y_train(N_train, y_shape);

  MatrixXf x_test(N_test, x_shape);
  MatrixXf y_test(N_test, y_shape);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

  std::vector<Layer*> layers;
  std::vector<Functor*> functors;
  std::vector<Component*> components;

  for (std::size_t i = 0; i < size - 1; ++i) {
    std::size_t n_input = i ? n_hidden : x_shape;
    layers.push_back(new HiddenLayer(n_input, n_hidden, n_output));
    functors.push_back(new LayerFunctor(layers.back()));
    components.push_back(new Component(*functors.back(), i));
  }

  layers.push_back(new OutputLayer(n_hidden, n_output));
  functors.push_back(new LayerFunctor(layers.back()));
  components.push_back(new Component(*functors.back(), size - 1));

  ErrorFunctor error_functor;
  Pipe pipe;

  Component error_component(error_functor, size - 1);
  Component image_pipe(pipe, 0);
  Component label_pipe(pipe, 0);

  for (std::size_t i = 0; i < components.size(); ++i) {
    Component* from = i ? components[i - 1] : &image_pipe;
    components[i]->connect(from);
    components[i]->connect(&error_component);
  }

  error_component.connect(components.back());
  error_component.connect(&label_pipe);

  VTSScheduler s(components);

  s.addComponent(&error_component);
  s.addComponent(&image_pipe);
  s.addComponent(&label_pipe);

  MT19937 rng;

  for (std::size_t epoch = 0; epoch < n_epoch; ++epoch) {
    shuffle(perm, perm + N_train, rng);

    Timer timer;
    for (std::size_t batchnum = 0; batchnum < N_train; batchnum += batchsize) {
      if (rank == 0) {
        if ((batchnum + batchsize) % 800 == 0) std::cerr << "#";

        MatrixXf x(batchsize, x_shape);
        MatrixXf t(batchsize, y_shape);

        for (std::size_t i = 0; i < batchsize; ++i) {
          x.row(i) = x_train.row(perm[batchnum + i]);
          t.row(i) = y_train.row(perm[batchnum + i]);
        }

        image_pipe.setInput(0, bufferFromMatrix(x));
        label_pipe.setInput(0, bufferFromMatrix(t));
      }

      s.step();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::cerr << std::endl;
      std::cout << epoch << " " << timer.elapsed() << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (std::size_t i = 0; i < layers.size() - 1; ++i) {
      if (i == rank) {
        HiddenLayer* layer = dynamic_cast<HiddenLayer*>(layers[i]);
        std::cout << "Loss: " << std::setprecision(7)
                  << layer->loss / layer->count << std::endl;
        layer->loss = 0.0;
        layer->count = 0;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == size - 1) {
      std::cout << "Loss: " << std::setprecision(7)
                << error_functor.loss / error_functor.count
                << " Accuracy: " << error_functor.acc / error_functor.count
                << std::endl;
      error_functor.loss = 0.0;
      error_functor.acc = 0.0;
      error_functor.count = 0;
    }
  }

  for (std::size_t i = 0; i < layers.size(); ++i) {
    delete layers[i];
  }

  for (std::size_t i = 0; i < functors.size(); ++i) {
    delete functors[i];
  }

  for (std::size_t i = 0; i < components.size(); ++i) {
    delete components[i];
  }

  MPI_Finalize();

  return 0;
}
