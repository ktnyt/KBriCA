#include <cmath>
#include <ctime>
#include <limits>
#include <utility>
#include "mt64.hpp"

template <typename T>
void shuffle(T* begin, T* end, MT19937 rng) {
  int range = end - begin;
  int i, j;

  for (i = range - 1; i > 0; --i) {
    j = static_cast<int>(rng.real<float>() * i);
    std::swap(begin[i], begin[j]);
  }
}

template <typename T>
class Uniform {
 public:
  Uniform(T min = 0.0, T max = 1.0) {}
  T operator()(MT19937 rng) { return rng.real<T>() * (max - min) + min; }

 private:
  T min, max;
};

template <typename T>
class Normal {
 public:
  Normal(T mu = 0.0, T sigma = 1.0) {}
  T operator()(MT19937 rng) {
    if (has_w) {
      has_w = false;
      return w * sigma + mu;
    }

    do {
      u = rng.real<T>(false) * 2 - 1.0;
      v = rng.real<T>(false) * 2 - 1.0;
      s = u * u + v * v;
    } while (s >= 1 || s == 0);

    m = sqrt(-2.0 * log(s) / s);
    w = v * m;
    has_w = true;
    return u * sigma + mu;
  }

 private:
  T mu, sigma, u, v, w, s, m;
  bool has_w;
};
