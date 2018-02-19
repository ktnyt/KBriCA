#ifndef __KBRICA_FUNCTOR_HPP__
#define __KBRICA_FUNCTOR_HPP__

#include <vector>
#include "kbrica/buffer.hpp"

namespace kbrica {

class Functor {
 public:
  virtual Buffer operator()(std::vector<Buffer>& inputs) = 0;
};

}  // namespace kbrica

#endif
