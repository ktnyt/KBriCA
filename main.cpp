#include <unistd.h>
#include <iostream>
#include <vector>

#include "mpi.h"

#include "kbrica.hpp"

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

class Load : public Functor {
 public:
  Load(int payload, int delay) : buffer(payload), delay(delay) {}
  Buffer operator()(std::vector<Buffer>& inputs) {
    Timer timer;
    while (timer.elapsed() < delay) {
    }
    return buffer;
  }

 private:
  Buffer buffer;
  int delay;
};

int main(int argc, char* argv[]) {
  int miniter = 10;
  int payload = 1000 * sizeof(float);
  int delay = 100;
  int n = 1024;

  MPI_Init(&argc, &argv);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (int procs = 1; procs <= size; ++procs) {
    std::vector<std::vector<Component*> > components(procs);
    std::vector<Component*> flattened(n);

    Load load(payload, delay);

    for (int i = 0; i < n; ++i) {
      Component* component = new Component(load, i % procs);
      components[i % procs].push_back(component);
      flattened[i] = component;
    }

    for (int i = 0; i < components.size(); ++i) {
      for (int j = 0; j < components[i].size() - 1; ++j) {
        int k = j + 1;
        std::vector<Component*> inputs;
        inputs.push_back(components[i][j]);
        components[i][k]->connect(inputs);
      }

      if (i > 0) {
        int j = i - 1;
        std::vector<Component*> inputs;
        inputs.push_back(components[i].front());
        components[j].back()->connect(inputs);
      }
    }

    VTSScheduler s(flattened);

    Timer timer;

    int iters = 0;

    while (timer.elapsed() < 1000 * 1000 || iters < miniter) {
      s.step();
      ++iters;
    }

    if (rank == 0) {
      int elapsed = timer.elapsed();
      std::cout << procs << " " << elapsed / iters << std::endl;
    }

    for (int i = 0; i < n; ++i) {
      flattened[i]->wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (std::size_t i = 0; i < flattened.size(); ++i) {
      delete flattened[i];
    }
  }

  MPI_Finalize();

  return 0;
}
