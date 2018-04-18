#ifndef __KBRICA_FUNCTOR_HPP__
#define __KBRICA_FUNCTOR_HPP__

#include <vector>
#include "kbrica/shared_vector.hpp"

namespace kbrica {

class Functor {
 public:
  virtual ~Functor() {}
  virtual Buffer operator()(std::vector<Buffer>& inputs) = 0;
};

}  // namespace kbrica

#endif
