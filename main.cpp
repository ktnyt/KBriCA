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
  int iters = 5;
  int payload = 1000 * sizeof(float);
  int delay = 100;
  int n = 1024;

  MPI_Init(&argc, &argv);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Component*> components(n);

  for (int procs = 1; procs <= size; ++procs) {
    int m = n / procs;

    Load load(payload, delay);

    for (int i = 0; i < n; ++i) {
      int group = i / m;
      components[i] = new Component(load, group);
    }

    for (int i = 0; i < n - 1; ++i) {
      int j = i + 1;
      Component* source = components[i];
      Component* target = components[j];
      std::vector<Component*> tmp;
      tmp.push_back(source);
      target->connect(tmp);
    }

    VTSScheduler s(components);

    Timer timer;

    for (int i = 0; i < iters * procs; ++i) {
      s.step();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      int elapsed = timer.elapsed();
      std::cout << procs << " " << elapsed / (iters * procs) << std::endl;
    }
  }

  for (std::size_t i = 0; i < components.size(); ++i) {
    delete components[i];
  }

  MPI_Finalize();

  return 0;
}
