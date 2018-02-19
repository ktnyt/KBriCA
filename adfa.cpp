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
#include "utils.hpp"

using namespace Eigen;
using namespace kbrica;

class HiddenLayer : public Functor {
 public:
  HiddenLayer(int n_input, int n_output, int n_final, float lr, float decay)
      : lr(lr),
        decay(decay),
        aelr(lr * 0.1),
        epsilon(std::numeric_limits<float>::epsilon()) {
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
    {
      B = MatrixXf::Random(n_final, n_output);
      float max = B.maxCoeff();
      B /= (max * sqrt(static_cast<float>(n_final)));
    }
  }

  Buffer operator()(std::vector<Buffer> inputs) {
    Buffer input = inputs[0];
    Buffer error = inputs[1];
    Buffer output(0);

    if (input.len() != 0) {
      char* buffer = input.get();

      int num = *reinterpret_cast<int*>(buffer);
      buffer += sizeof(int);

      int rows = *reinterpret_cast<int*>(buffer);
      buffer += sizeof(int);

      int cols = *reinterpret_cast<int*>(buffer);
      buffer += sizeof(int);

      float* data = reinterpret_cast<float*>(buffer);

      Map<MatrixXf> x(data, rows, cols);
      MatrixXf y = (x * W).unaryExpr(&sigmoid);

      if (aelr > epsilon) {
        MatrixXf z = (y * U).unaryExpr(&sigmoid);

        MatrixXf d_z = z - x;
        MatrixXf d_y =
            (d_z * U.transpose()).array() * y.unaryExpr(&dsigmoid).array();

        MatrixXf d_W = -x.transpose() * d_y;
        MatrixXf d_U = -y.transpose() * d_z;

        W += d_W * aelr;
        U += d_U * aelr;

        aelr *= decay;
      }

      int size = sizeof(int) * 3 + sizeof(float) * y.size();
      output = Buffer(size);
      std::copy(y.data(), y.data() + y.size(), output.get());

      input_queue.push(input);
      output_queue.push(output);
    }

    if (error.len() != 0) {
      Buffer input = input_queue.front();
      Buffer output = output_queue.front();
      input_queue.pop();
      output_queue.pop();
    }

    return output;
  }

 private:
  MatrixXf W;
  MatrixXf U;
  MatrixXf B;

  std::queue<Buffer> input_queue;
  std::queue<Buffer> output_queue;

  float lr;
  float decay;
  float aelr;
  float epsilon;
};

class Pipe : public Functor {
 public:
  Buffer operator()(std::vector<Buffer>& inputs) {
    if (inputs[0].len() == 0) {
      return Buffer(0);
    }
    return inputs[0].clone();
  }
};

template <typename T>
class ImageLoader : public Functor {
 public:
  ImageLoader(MNIST<T>& mnist) : mnist(mnist) {}
  Buffer operator()(std::vector<Buffer>& inputs) {
    T* images = mnist.train_images[batch];
    char* casted = reinterpret_cast<char*>(images);

    Buffer buffer(mnist.train_images.dims * sizeof(T));

    std::copy(casted, casted + buffer.len(), buffer.get());

    batch += 1;
    batch %= mnist.train_images.length;

    return buffer;
  }

 private:
  MNIST<T>& mnist;
  int batch;
};

class Tester : public Functor {
 public:
  Tester() : output(sizeof(int)) { *reinterpret_cast<int*>(output.get()) = 0; }
  Buffer operator()(std::vector<Buffer>& inputs) {
    Buffer x = inputs[0];
    Buffer y = inputs[1];

    if (x.len()) {
      x_list.push_back(x);
    }

    if (y.len()) {
      y_list.push_back(y);
    }

    while (x_list.size() && y_list.size()) {
      std::cout << "check" << std::endl;
      Buffer a = x_list.front();
      Buffer b = y_list.front();
      x_list.pop_front();
      y_list.pop_front();
      print_mnist(reinterpret_cast<int*>(a.get()));
      print_mnist(reinterpret_cast<int*>(b.get()));
    }

    return output;
  }

 private:
  Buffer output;
  std::list<Buffer> x_list;
  std::list<Buffer> y_list;
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MNIST<int> mnist("./mnist");
  mnist.train_images.scale(64, true);

  ImageLoader<int> loader(mnist);
  Pipe pipe1;
  Pipe pipe2;
  Tester tester;

  Component* c0 = new Component(loader, 0);
  Component* c1 = new Component(pipe1, 1);
  Component* c2 = new Component(pipe2, 2);
  Component* c3 = new Component(tester, 3);

  {
    std::vector<Component*> inputs;
    inputs.push_back(c0);
    c1->connect(inputs);
  }

  {
    std::vector<Component*> inputs;
    inputs.push_back(c1);
    c2->connect(inputs);
  }

  {
    std::vector<Component*> inputs;
    inputs.push_back(c2);
    inputs.push_back(c0);
    c3->connect(inputs);
  }

  std::vector<Component*> all;
  all.push_back(c0);
  all.push_back(c1);
  all.push_back(c2);
  all.push_back(c3);

  VTSScheduler s(all);

  s.step();
  s.step();
  s.step();
  s.step();

  delete c0;
  delete c1;
  delete c2;
  delete c3;

  MPI_Finalize();

  return 0;
}
