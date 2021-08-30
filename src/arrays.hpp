#pragma once

//! \file arrays.hpp
//! \brief Helper methods for multi-dimensional arrays

#include <vector>

namespace calibr8 {

//! \brief A 1-dimensional array type
//! \tparam The underlying type of the array
template <typename T>
using Array1D = std::vector<T>;

//! \brief A 2-dimensional array type
//! \tparam The underlying type of the array
template <typename T>
using Array2D = std::vector<std::vector<T>>;

//! \brief A 3-dimensional array type
//! \tparam The underlying type of the array
template <typename T>
using Array3D = std::vector<std::vector<std::vector<T>>>;

//! \brief Resize a 1D array type
//! \param a The array to resize
//! \param ni The number of entries in the first dimension
template <typename T>
void resize(Array1D<T>& a, int ni) {
  a.resize(ni);
}

//! \brief Resize a 2D array type
//! \param a The array to resize
//! \param ni The number of entries in the first dimension
//! \param nj The number of entries in the second dimension
template <typename T>
void resize(Array2D<T>& a, int ni, int nj) {
  a.resize(ni);
  for (int i = 0; i < ni; ++i) {
    a[i].resize(nj);
  }
}

//! \brief Resize a 2D array type
//! \param a The array to resize
//! \param ni The number of entries in the first dimension
//! \param nj The number of entries in the second dimension as a function of i
template <typename T>
void resize(Array2D<T>& a, int ni, Array1D<int> const& nj) {
  a.resize(ni);
  for (int i = 0; i < ni; ++i) {
    a[i].resize(nj[i]);
  }
}

//! \brief resize the 3D array type
//! \tparam The underlying type of the array
//! \param a The array to resize
//! \param ni The number of entries in the first dimension
//! \param nj The number of entries in the second dimension as a function of i
//! \param nk The number of entries in the third dimension
template <typename T>
void resize(Array3D<T>& a, int ni, Array1D<int> const& nj, int nk) {
  a.resize(ni);
  for (int i = 0; i < ni; ++i) {
    a[i].resize(nj[i]);
    for (int j = 0; j < nj[i]; ++j) {
      a[i][j].resize(nk);
    }
  }
}

//! \brief resize the 3D array type
//! \tparam The underlying type of the array
//! \param a The array to resize
//! \param ni The number of entries in the first dimension
//! \param nj The number of entries in the second dimension
//! \param nk The number of entries in the third dimension as a function of i
template <typename T>
void resize(Array3D<T>& a, int ni, int nj, Array1D<int> const& nk) {
  a.resize(ni);
  for (int i = 0; i < ni; ++i) {
    a[i].resize(nj);
    for (int j = 0; j < nj; ++j) {
      a[i][j].resize(nk[i]);
    }
  }
}

}
