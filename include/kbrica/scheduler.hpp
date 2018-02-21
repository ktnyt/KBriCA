#ifndef __KBRICA_SCHEDULER_HPP__
#define __KBRICA_SCHEDULER_HPP__

#include <vector>

#include "kbrica/component.hpp"

#include "mpi.h"

namespace kbrica {

class VTSScheduler {
 public:
  VTSScheduler() {}
  VTSScheduler(std::vector<Component*> components) : components_(components) {}

  void addComponent(Component* component) { components_.push_back(component); }

  void step() {
    for (std::size_t i = 0; i < components_.size(); ++i) {
      components_[i]->collect();
    }

    for (std::size_t i = 0; i < components_.size(); ++i) {
      components_[i]->execute();
    }

    for (std::size_t i = 0; i < components_.size(); ++i) {
      components_[i]->expose();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

 private:
  std::vector<Component*> components_;
};

}  // namespace kbrica

#endif
