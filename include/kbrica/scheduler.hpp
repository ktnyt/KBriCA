#ifndef __KBRICA_SCHEDULER_HPP__
#define __KBRICA_SCHEDULER_HPP__

#include <vector>

#include "kbrica/component.hpp"

#include "mpi.h"

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

namespace kbrica {

class VTSScheduler {
 public:
  VTSScheduler(std::vector<Component*> components) : components_(components) {}

  void step() {
#pragma omp for nowait
    for (std::size_t i = 0; i < components_.size(); ++i) {
      components_[i]->collect();
    }

#pragma omp for nowait
    for (std::size_t i = 0; i < components_.size(); ++i) {
      components_[i]->execute();
    }

#pragma omp for nowait
    for (std::size_t i = 0; i < components_.size(); ++i) {
      components_[i]->expose();
    }

#pragma omp barrier
    MPI_Barrier(MPI_COMM_WORLD);
  }

 private:
  std::vector<Component*> components_;
};

}  // namespace kbrica

#endif
