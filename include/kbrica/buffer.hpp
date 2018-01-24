#ifndef __KBRICA_BUFFER_HPP__
#define __KBRICA_BUFFER_HPP__

#include <algorithm>
#include <utility>

namespace kbrica {

class Buffer {
 public:
  Buffer(int size = 0)
      : buffer(new char[sizeof(int) + size]), count(new std::size_t(1)) {
    *reinterpret_cast<int*>(buffer) = sizeof(int) + size;
  }

  Buffer(const Buffer& other) : buffer(other.buffer), count(other.count) {
    if (count) {
      ++*count;
    }
  }

  ~Buffer() { clear(); }

  Buffer& operator=(const Buffer& other) {
    if (this != &other) {
      clear();
      buffer = other.buffer;
      count = other.count;
      if (count) {
        ++*count;
      }
    }
    return *this;
  }

  char* get() const { return buffer + sizeof(int); }
  const int len() const { return size() - sizeof(int); }

  char* data() const { return buffer; }
  const int size() const { return *reinterpret_cast<int*>(buffer); }

  Buffer clone() {
    Buffer other(size());
    std::copy(buffer, buffer + size(), other.buffer);
    return other;
  }

 protected:
  void clear() {
    if (count) {
      if (*count == 1) delete[] buffer;
      if (!--*count) delete count;
    }
    buffer = NULL;
    count = NULL;
  }

 private:
  char* buffer;
  std::size_t* count;
};

}  // namespace kbrica

#endif
