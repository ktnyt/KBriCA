#include <algorithm>
#include <iostream>
#include <limits>
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

      int data = *reinterpret_cast<float*>(buffer);

      Map<MatrixXf> x(data, rows, cols);
      MatrixXf y = (x * W).unaryExpr(&sigmoid);

      if (aelr > epsilon) {
        MatrixXf z = (y * U).unaryExpr(&sigmoid);

        MatrixXf d_z = z - x;
        MatrixXf d_y =
            (d_z * U.transpose().array() * y.unaryExpr(&dsigmoid).array());

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

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Finalize();

  return 0;
}
