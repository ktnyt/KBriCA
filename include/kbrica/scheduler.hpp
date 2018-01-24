#ifndef __KBRICA_SCHEDULER_HPP__
#define __KBRICA_SCHEDULER_HPP__

#include <vector>

#include "kbrica/component.hpp"

namespace kbrica {

class VTSScheduler {
 public:
  VTSScheduler(std::vector<Component*> components) : components_(components) {}

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
  }

 private:
  std::vector<Component*> components_;
};

}  // namespace kbrica

#endif
