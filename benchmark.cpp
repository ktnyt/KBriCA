#include <unistd.h>
#include <iostream>
#include <vector>

#include <omp.h>
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
  Load(int payload, int delay) : payload(payload), delay(delay) {}
  Buffer operator()(std::vector<Buffer>& inputs) {
    Timer timer;
    Buffer buffer(payload);
    float value = 0.0;
    while (timer.elapsed() < delay) {
      for (int i = 0; i < payload / sizeof(float); ++i) {
        float tmp = (float)rand() / (float)(RAND_MAX);
        if (i % 2) {
          reinterpret_cast<float*>(buffer.get())[i] += value + tmp;
        } else {
          reinterpret_cast<float*>(buffer.get())[i] += value - tmp;
        }
      }
    }
    return buffer;
  }
  int output_size() { return payload; }

 private:
  int payload;
  int delay;
};

void run(int procs, int delay, int payload, int n) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

  MPI_Barrier(MPI_COMM_WORLD);

  VTSScheduler s(flattened);

  Timer timer;

  int iters = 0;

  while (iters < 100) {
    s.step();
    ++iters;
  }

  if (rank == 0) {
    int elapsed = timer.elapsed();
    std::cout << delay << " " << procs << " " << elapsed / iters / procs
              << std::endl;
  }

  for (int i = 0; i < n; ++i) {
    flattened[i]->wait();
  }

  for (std::size_t i = 0; i < flattened.size(); ++i) {
    delete flattened[i];
  }
}

int main(int argc, char* argv[]) {
  int delay = 100;
  int payload = 1000 * sizeof(float);

  MPI_Init(&argc, &argv);

  int size;

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int i = 0; i < 3; ++i) {
    for (int procs = 1; procs <= size; ++procs) {
      int n = procs * omp_get_max_threads();
      run(procs, delay, payload, n);
    }
    delay *= 10;
  }

  MPI_Finalize();

  return 0;
}
