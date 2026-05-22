#pragma once

#include <vector>

#include "types.h++"
#include "vector.h++"

template <typename T>
void write_npy_file(const char file_path[], const std::vector<T> &array);
