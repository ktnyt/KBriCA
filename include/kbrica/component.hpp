#ifndef __KBRICA_FUNCTOR_COMPONENT_HPP__
#define __KBRICA_FUNCTOR_COMPONENT_HPP__

#include "kbrica/functor.hpp"

#include "mpi.h"

namespace kbrica {

class Component {
 public:
  Component(Functor& f, int wanted, int tag = 0)
      : f(f), wanted(wanted), request(MPI_REQUEST_NULL), tag(tag), size(0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &actual);
  }

  void collect() {
    if (wanted == actual) {
      for (std::size_t i = 0; i < connected.size(); ++i) {
        inputs[i] = connected[i]->output;
      }
    }
  }

  void execute() {
    if (wanted == actual) {
      output = f(inputs);
    }
  }

  void expose() {
    if (request != MPI_REQUEST_NULL) {
      int flag;
      MPI_Test(&request, &flag, &status);

      if (!flag) {
        return;
      }

      request = MPI_REQUEST_NULL;
    }

    for (std::size_t i = 0; i < targets.size(); ++i) {
      if (wanted == actual && targets[i] != wanted) {
        size = output.size();
        MPI_Send(&size, 1, MPI_INT, targets[i], tag * 2 + 1, MPI_COMM_WORLD);
      }

      if (wanted != actual && targets[i] == actual) {
        MPI_Recv(&size, 1, MPI_INT, wanted, tag * 2 + 1, MPI_COMM_WORLD,
                 &status);
      }
    }

    for (std::size_t i = 0; i < targets.size(); ++i) {
      if (wanted == actual && targets[i] != wanted) {
        MPI_Isend(output.data(), output.size(), MPI_CHAR, targets[i], tag * 2,
                  MPI_COMM_WORLD, &request);
      }

      if (wanted != actual && targets[i] == actual) {
        if (output.size() != size) {
          output = Buffer(size);
        }

        MPI_Irecv(output.data(), output.size(), MPI_CHAR, wanted, tag * 2,
                  MPI_COMM_WORLD, &request);
      }
    }
  }

  void connect(std::vector<Component*> sources) {
    connected = sources;
    inputs = std::vector<Buffer>(connected.size());
    for (std::size_t i = 0; i < connected.size(); ++i) {
      connected[i]->addTarget(wanted);
    }
  }

  void addTarget(int rank) { targets.push_back(rank); }

  void wait() { MPI_Wait(&request, &status); }

  const Buffer getInput(std::size_t i) { return inputs[i]; }
  const Buffer getOutput() const { return output; }

 private:
  Functor& f;
  int wanted;
  int actual;
  std::vector<int> targets;

  MPI_Request request;
  MPI_Status status;

  int tag;

  std::vector<Buffer> inputs;
  Buffer output;
  int size;

  std::vector<Component*> connected;
};

}  // namespace kbrica

#endif
