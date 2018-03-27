#ifndef __KBRICA_BUFFER_HPP__
#define __KBRICA_BUFFER_HPP__

#include <algorithm>
#include <utility>

namespace kbrica {

template <class T>
class SharedVector {
 public:
  SharedVector(std::size_t size = 0)
      : data_(new T[size]), size_(size), count_(new std::size_t(1)) {}
  SharedVector(const SharedVector& other)
      : data_(other.data_), size_(other.size_), count_(other.count_) {
    ++*count_;
  }
  virtual ~SharedVector() { clear(); }

  SharedVector& operator=(const SharedVector& other) {
    if (this != &other) {
      clear();
      data_ = other.data_;
      size_ = other.size_;
      count_ = other.count_;
      ++*count_;
    }
  }

  std::size_t size() const { return size_; }
  std::size_t count() const { return count_; }
  void resize(std::size_t size) {
    if (size == size_) return;
    std::size_t copy_len = size > size_ ? size_ : size;
    size_ = size;
    T* next = new T[size_];
    std::copy(data_, data_ + copy_len, next);
    std::swap(data_, next);
    delete[] next;
  }
  T* data() { return data_; }

 private:
  void clear() {
    if (count_) {
      if (!--*count_) {
        delete[] data_;
        delete count_;
      }
    }
  }

  T* data_;
  std::size_t size_;
  std::size_t* count_;
};

typedef SharedVector<char> Buffer;

}  // namespace kbrica

#endif
