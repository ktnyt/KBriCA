#include <cmath>
#include <ctime>
#include <limits>
#include <utility>
#include "mt64.hpp"

static MT19937 rng__;

template <typename T>
void shuffle(T* begin, T* end) {
  int range = end - begin;
  int i, j;

  for (i = range - 1; i > 0; --i) {
    j = static_cast<int>(rng__.real<float>() * i);
    std::swap(begin[i], begin[j]);
  }
}

template <typename T>
class Uniform {
 public:
  Uniform(T min = 0.0, T max = 1.0) {}
  T operator()() { return mt.real<T>() * (max - min) + min; }

 private:
  MT19937 mt;
  T min, max;
};

template <typename T>
class Normal {
 public:
  Normal(T mu = 0.0, T sigma = 1.0) {}
  T operator()() {
    if (has_w) {
      has_w = false;
      return w * sigma + mu;
    }

    do {
      u = mt.real<T>(false) * 2 - 1.0;
      v = mt.real<T>(false) * 2 - 1.0;
      s = u * u + v * v;
    } while (s >= 1 || s == 0);

    m = sqrt(-2.0 * log(s) / s);
    w = v * m;
    has_w = true;
    return u * sigma + mu;
  }

 private:
  MT19937 mt;
  T mu, sigma, u, v, w, s, m;
  bool has_w;
};
