#include <fstream>
#include <vector>

int reverse_int(int i) {
  unsigned char i0, i1, i2, i3;
  i0 = i & 255;
  i1 = (i >> 8) & 255;
  i2 = (i >> 16) & 255;
  i3 = (i >> 24) & 255;
  return (static_cast<int>(i0) << 24) + (static_cast<int>(i1) << 16) +
         (static_cast<int>(i2) << 8) + static_cast<int>(i3);
}

std::vector<std::vector<unsigned char> > read_images(const char* path) {
  std::vector<std::vector<unsigned char> > array;
  std::ifstream file(path, std::ios::binary);
  bool reverse = false;
  if (file.is_open()) {
    int magic_number;
    int n_images;
    int n_rows;
    int n_cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 2051) {
      if (reverse_int(magic_number) != 2051) {
        return array;
      }
      magic_number = reverse_int(magic_number);
      reverse = true;
    }
    file.read(reinterpret_cast<char*>(&n_images), sizeof(n_images));
    if (reverse) {
      n_images = reverse_int(n_images);
    }
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    if (reverse) {
      n_rows = reverse_int(n_rows);
    }
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    if (reverse) {
      n_cols = reverse_int(n_cols);
    }
    array.resize(n_images);
    for (int i = 0; i < n_images; ++i) {
      array[i].resize(n_rows * n_cols);
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char tmp;
          file.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));
          array[i][n_rows * r + c] = tmp;
        }
      }
    }
  }
  return array;
}

std::vector<unsigned char> read_labels(const char* path) {
  std::vector<unsigned char> array;
  std::ifstream file(path, std::ios::binary);
  if (file.is_open()) {
    int magic_number;
    int n_labels;
    bool reverse = false;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 2049) {
      if (reverse_int(magic_number) != 2049) {
        return array;
      }
      magic_number = reverse_int(magic_number);
      reverse = true;
    }
    magic_number = reverse_int(magic_number);
    file.read(reinterpret_cast<char*>(&n_labels), sizeof(n_labels));
    if (reverse) {
      n_labels = reverse_int(n_labels);
    }
    array.resize(n_labels);
    for (int i = 0; i < n_labels; ++i) {
      unsigned char tmp;
      file.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));
      array[i] = tmp;
    }
  }
  return array;
}
