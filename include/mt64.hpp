#ifndef __MT64_HPP__
#define __MT64_HPP__

#include <ctime>

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL         /* Least significant 31 bits */
#define SM 6364136223846793005ULL

class MT19937 {
 public:
  MT19937(unsigned long long seed = time(NULL)) {
    mag01[0] = 0ULL;
    mag01[1] = MATRIX_A;

    mt[0] = seed;
    for (mti = 1; mti < NN; ++mti) {
      mt[mti] = (SM * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti);
    }
  }

  unsigned long long integer() {
    int i;
    unsigned long long x;

    if (mti >= NN) {
      for (i = 0; i < NN - MM; ++i) {
        x = (mt[i] & UM) | (mt[i + 1] & LM);
        mt[i] = mt[i + MM] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1)];
      }

      for (; i < NN - 1; ++i) {
        x = (mt[i] & UM) | (mt[i + 1] & LM);
        mt[i] = mt[i + (MM - NN)] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1)];
      }

      x = (mt[NN - 1] & UM) | (mt[0] & LM);
      mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ mag01[static_cast<int>(x & 1)];

      mti = 0;
    }

    x = mt[mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
  }

  template <typename T>
  T real(bool inclusive = true) {
    if (inclusive) {
      return (integer() >> 11) * (1.0 / 9007199254740992.0);
    }

    return ((integer() >> 12) + 0.5) * (1.0 / 4503599627370496.0);
  }

 private:
  unsigned long long mt[NN];
  int mti;

  unsigned long long mag01[2];
};

#endif
