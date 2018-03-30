#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <vector>

#include "Eigen/Core"
#include "kbrica.hpp"
#include "mpi.h"

#include "activations.hpp"
#include "functions.hpp"
#include "random.hpp"
#include "utils.hpp"

using namespace Eigen;
using namespace kbrica;

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

static MT19937 engine;

MatrixXf linear(MatrixXf& x, MatrixXf& W, VectorXf& b) {
  MatrixXf a = x * W;
  a.transpose().colwise() += b;
  return a;
}

class DFALayer : public Functor {
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
        count(0),
        epsilon(std::numeric_limits<float>::epsilon()) {
    float stdW = 1. / sqrt(static_cast<float>(n_input));
    float stdB = 1. / sqrt(static_cast<float>(n_output));

    Normal<float> genW(0.0, stdW);
    Normal<float> genB(0.0, stdB);

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
      MatrixXf y = (x * W).unaryExpr(&sigmoid);

      xs.push(x);
      ys.push(y);

      ret = bufferFromMatrix(y);

      if (ae > epsilon) {
        MatrixXf z = (y * W.transpose()).unaryExpr(&sigmoid);

        MatrixXf d_z = z - x;
        MatrixXf d_y = (d_z * W).array() * y.unaryExpr(&dsigmoid).array();

        MatrixXf d_W = -(x.transpose() * d_y + d_z.transpose() * y);

        W += d_W * ae;

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

 private:
  float epsilon;
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

    Normal<float> genW(0.0, stdW);

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

int reverse_int(int i) {
  unsigned char i0, i1, i2, i3;
  i0 = i & 255;
  i1 = (i >> 8) & 255;
  i2 = (i >> 16) & 255;
  i3 = (i >> 24) & 255;
  return (static_cast<int>(i0) << 24) + (static_cast<int>(i1) << 16) +
         (static_cast<int>(i2) << 8) + static_cast<int>(i3);
}

std::vector<std::vector<unsigned char> > read_image(const char* path) {
  std::vector<std::vector<unsigned char> > array;
  std::ifstream file(path, std::ios::binary);
  bool reverse = false;
  if (file.is_open()) {
    int magic_number;
    int n_images;
    int n_rows;
    int n_cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 2051) {
      if (reverse_int(magic_number) != 2051) {
        return array;
      }
      magic_number = reverse_int(magic_number);
      reverse = true;
    }
    file.read(reinterpret_cast<char*>(&n_images), sizeof(n_images));
    if (reverse) {
      n_images = reverse_int(n_images);
    }
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    if (reverse) {
      n_rows = reverse_int(n_rows);
    }
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    if (reverse) {
      n_cols = reverse_int(n_cols);
    }
    array.resize(n_images);
    for (int i = 0; i < n_images; ++i) {
      array[i].resize(n_rows * n_cols);
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char tmp;
          file.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));
          array[i][n_rows * r + c] = tmp;
        }
      }
    }
  }
  return array;
};

std::vector<unsigned char> read_label(const char* path) {
  std::vector<unsigned char> array;
  std::ifstream file(path, std::ios::binary);
  if (file.is_open()) {
    int magic_number;
    int n_labels;
    bool reverse = false;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 2049) {
      if (reverse_int(magic_number) != 2049) {
        return array;
      }
      magic_number = reverse_int(magic_number);
      reverse = true;
    }
    magic_number = reverse_int(magic_number);
    file.read(reinterpret_cast<char*>(&n_labels), sizeof(n_labels));
    if (reverse) {
      n_labels = reverse_int(n_labels);
    }
    array.resize(n_labels);
    for (int i = 0; i < n_labels; ++i) {
      unsigned char tmp;
      file.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));
      array[i] = tmp;
    }
  }
  return array;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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

  Timer timer;

  for (int n_procs = 1; n_procs < size; ++n_procs) {
    int n = n_procs + 1;
    if ((n & (n - 1)) != 0) {
      continue;
    }

    Pipe pipe;

    std::vector<DFALayer> hidden_layers;
    DFALayer layer(x_shape, n_hidden, 10, 0.01, 0.001, 0.9995);
    hidden_layers.push_back(layer);
    for (int i = 1; i < n_procs - 1; ++i) {
      DFALayer layer(n_hidden, n_hidden, 10, 0.01, 0.001, 0.9995);
      hidden_layers.push_back(layer);
    }
    OutLayer output_layer(n_hidden, 10, 0.01);

    std::vector<Component*> components;
    Component* image_pipe = new Component(pipe, 0);
    Component* label_pipe = new Component(pipe, 0);
    Component* input_component = new Component(hidden_layers[0], 0);
    Component* output_component = new Component(output_layer, n_procs);

    input_component->connect(image_pipe);
    input_component->connect(output_component);

    components.push_back(input_component);

    for (int i = 1; i < n_procs - 1; ++i) {
      Component* component = new Component(hidden_layers[i], i);
      components.push_back(component);

      component->connect(components[i - 1]);
      component->connect(output_component);
    }

    output_component->connect(components.back());
    output_component->connect(label_pipe);

    components.push_back(output_component);

    components.push_back(image_pipe);
    components.push_back(label_pipe);

    VTSScheduler s(components);

    MT19937 rng;

    timer.reset();

    for (std::size_t epoch = 0; epoch < n_epoch; ++epoch) {
      if (rank == 0) {
        std::cerr << "Epoch: " << epoch << std::endl;
        shuffle(perm, perm + N_train, rng);
      }

      for (std::size_t batchnum = 0; batchnum < N_train;
           batchnum += batchsize) {
        if (rank == 0) {
          if ((batchnum + batchsize) % 800 == 0) std::cerr << "#";

          MatrixXf x(batchsize, x_shape);
          MatrixXf t(batchsize, y_shape);

          for (std::size_t i = 0; i < batchsize; ++i) {
            x.row(i) = x_train.row(perm[batchnum + i]);
            t.row(i) = y_train.row(perm[batchnum + i]);
          }

          image_pipe->setInput(0, bufferFromMatrix(x));
          label_pipe->setInput(0, bufferFromMatrix(t));
        }

        s.step();
      }

      MPI_Barrier(MPI_COMM_WORLD);

      if (rank == 0) {
        std::cerr << std::endl;
      }

      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < hidden_layers.size(); ++i) {
        if (hidden_layers[i].count) {
          std::cerr << "Train\tLayer " << i << "\t"
                    << "Loss: " << std::setprecision(7)
                    << hidden_layers[i].loss / hidden_layers[i].count
                    << std::endl;
          hidden_layers[i].loss = 0.0;
          hidden_layers[i].count = 0;
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }

      if (rank == n_procs - 1) {
        std::cerr << "Train\t Output Loss: " << std::setprecision(7)
                  << output_layer.loss / output_layer.count
                  << " Accuracy: " << output_layer.acc / output_layer.count
                  << std::endl;
        output_layer.loss = 0.0;
        output_layer.acc = 0.0;
        output_layer.count = 0;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << n_procs << "\t" << timer.elapsed() << std::endl;
    }

    for (int i = 0; i < components.size(); ++i) {
      delete components[i];
    }
  }

  MPI_Finalize();

  return 0;
}
