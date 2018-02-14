#include <unistd.h>
#include <iostream>
#include <vector>

#include "mpi.h"

#include "kbrica.hpp"

using namespace kbrica;

class Load : public Functor {
 public:
  Load(int payload, int delay) : payload(payload), delay(delay) {}
  Buffer operator()(std::vector<Buffer>& inputs) {
    usleep(delay);
    return Buffer(payload);
  }

 private:
  int payload;
  int delay;
};

int main(int argc, char* argv[]) {
  int iters = 100;
  int payload = 1000;
  int delay = 10;
  int n = 1000;

  MPI_Init(&argc, &argv);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int m = n / size;

  std::vector<Component*> components(n);

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

  struct timespec start, finish;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < iters; ++i) {
    s.step();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec) * 1000;
  elapsed += (finish.tv_nsec - start.tv_nsec) / (1000 * 1000);

  if (rank == 0) {
    std::cout << elapsed << std::endl;
  }

  for (std::size_t i = 0; i < components.size(); ++i) {
    delete components[i];
  }

  MPI_Finalize();

  return 0;
}
