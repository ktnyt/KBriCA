#ifndef __KBRICA_SHARED_VECTOR_HPP__
#define __KBRICA_SHARED_VECTOR_HPP__

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace kbrica {

template <class T>
class SharedVector {
  using V = std::vector<T>;

 public:
  using reference = typename V::reference;
  using const_reference = typename V::const_reference;
  using iterator = typename V::iterator;
  using const_iterator = typename V::const_iterator;
  using size_type = typename V::size_type;
  using difference_type = typename V::difference_type;
  using value_type = typename V::value_type;
  using allocator_type = typename V::allocator_type;
  using pointer = typename V::pointer;
  using const_pointer = typename V::const_pointer;
  using reverse_iterator = typename V::reverse_iterator;
  using const_reverse_iterator = typename V::const_reverse_iterator;

  SharedVector(int size = 0) : ptr(std::make_shared<V>(size)) {}
  SharedVector(const SharedVector& other)
      : ptr(std::make_shared<V>(*other.ptr.get())) {}
  SharedVector(SharedVector&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  ~SharedVector() {}

  SharedVector& operator=(const SharedVector& other) {
    SharedVector another(other);
    *this = std::move(another);
    return *this;
  }

  SharedVector& operator=(SharedVector&& other) noexcept {
    using std::swap;
    swap(*this, other);
    return *this;
  }

  SharedVector clone() {
    SharedVector other(size());
    std::copy(*ptr.get()->begin(), *ptr.get()->end(), other.ptr.get()->begin());
    return other;
  }

  iterator begin() noexcept { return ptr.get()->begin(); }
  iterator end() noexcept { return ptr.get()->end(); }
  const_iterator begin() const noexcept { return ptr.get()->begin(); }
  const_iterator end() const noexcept { return ptr.get()->end(); }
  const_iterator cbegin() const noexcept { return ptr.get()->cbegin(); }
  const_iterator cend() const noexcept { return ptr.get()->cend(); }

  reverse_iterator rbegin() noexcept { return ptr.get()->rbegin(); }
  reverse_iterator rend() noexcept { return ptr.get()->rend(); }
  const_reverse_iterator rbegin() const noexcept { return ptr.get()->rbegin(); }
  const_reverse_iterator rend() const noexcept { return ptr.get()->rend(); }
  const_reverse_iterator crbegin() const noexcept {
    return ptr.get()->crbegin();
  }
  const_reverse_iterator crend() const noexcept { return ptr.get()->crend(); }

  std::size_t size() const { return ptr.get()->size(); }
  std::size_t max_size() const { return ptr.get()->max_size(); }
  void resize(std::size_t sz) { ptr.get()->resize(sz); }
  void resize(std::size_t sz, const_reference c) { ptr.get()->resize(sz, c); }
  void resize(std::size_t sz, T c = T()) { ptr.get()->resize(sz, c); }
  std::size_t capacity() const { return ptr.get()->capacity(); }
  bool empty() const noexcept { return ptr.get()->empty(); }
  void reserve(std::size_t n) { ptr.get()->reserve(n); }
  void shrink_to_fit() { ptr.get()->shrink_to_fit(); }

  reference operator[](std::size_t n) { return ptr.get()->operator[](n); }
  const_reference operator[](std::size_t n) const {
    return ptr.get()->operator[](n);
  }
  reference at(std::size_t n) { return ptr.get()->at(n); }
  const_reference at(std::size_t n) const { return ptr.get()->at(n); }
  T* data() noexcept { return ptr.get()->data(); }
  T* data() const noexcept { return ptr.get()->data(); }
  reference front() { return ptr.get()->front(); }
  const_reference front() const { return ptr.get()->front(); }
  reference back() { return ptr.get()->back(); }
  const_reference back() const { return ptr.get()->back(); }

  template <class InputIterator>
  void assign(InputIterator first, InputIterator last) {
    ptr.get()->assign(first, last);
  }
  void assign(std::size_t n, const_reference u) { ptr.get()->assign(n, u); }
  void assign(std::initializer_list<T> l) { ptr.get()->assign(l); }
  void push_back(reference x) { ptr.get()->push_back(x); }
  void push_back(const_reference x) { ptr.get()->push_back(x); }
  template <class... Args>
  void emplace_back(Args&&... args) {
    ptr.get()->emplace_back(args...);
  }
  void pop_back() { ptr.get()->pop_back(); }
  iterator insert(iterator position, const_reference x) {
    return ptr.get()->insert(position, x);
  }
  iterator insert(const_iterator position, const_reference x) {
    return ptr.get()->insert(position, x);
  }
  iterator insert(const_iterator position, reference& x) {
    return ptr.get()->insert(position, x);
  }
  void insert(iterator position, std::size_t n, reference& x) {
    ptr.get()->insert(position, n, x);
  }
  template <class InputIterator>
  void insert(iterator position, InputIterator first, InputIterator last) {
    ptr.get()->insert(position, first, last);
  }
  template <class InputIterator>
  iterator insert(const_iterator position, InputIterator first,
                  InputIterator last) {
    return ptr.get()->insert(position, first, last);
  }
  iterator insert(const_iterator position, std::initializer_list<T> il) {
    return ptr.get()->insert(position, il);
  }
  template <class... Args>
  iterator emplace(const_iterator position, Args&&... args) {
    return ptr.get()->emplace(position, args...);
  }
  iterator erase(iterator position) { return ptr.get()->erase(position); }
  iterator erase(const_iterator position) { return ptr.get()->erase(position); }
  iterator erase(iterator first, iterator last) {
    return ptr.get()->erase(first, last);
  }
  iterator erase(const_iterator first, const_iterator last) {
    return ptr.get()->erase(first, last);
  }

  allocator_type get_allocator() const noexcept {
    return ptr.get()->get_allocator();
  }

  template <class U>
  friend bool operator==(const SharedVector<U>& x, const SharedVector<U>& y) {
    return *(x.ptr.get()) == *(y.ptr.get());
  }
  template <class U>
  friend bool operator!=(const SharedVector<U>& x, const SharedVector<U>& y) {
    return *(x.ptr.get()) != *(y.ptr.get());
  }
  template <class U>
  friend bool operator<(const SharedVector<U>& x, const SharedVector<U>& y) {
    return *(x.ptr.get()) < *(y.ptr.get());
  }
  template <class U>
  friend bool operator<=(const SharedVector<U>& x, const SharedVector<U>& y) {
    return *(x.ptr.get()) <= *(y.ptr.get());
  }
  template <class U>
  friend bool operator>(const SharedVector<U>& x, const SharedVector<U>& y) {
    return *(x.ptr.get()) > *(y.ptr.get());
  }
  template <class U>
  friend bool operator>=(const SharedVector<U>& x, const SharedVector<U>& y) {
    return *(x.ptr.get()) >= *(y.ptr.get());
  }
  template <class U>
  friend void swap(SharedVector<U>& a, SharedVector<U>& b) {
    std::swap(a.ptr, b.ptr);
  }

 private:
  std::shared_ptr<V> ptr;
};

using Buffer = SharedVector<char>;

}  // namespace kbrica

#endif
