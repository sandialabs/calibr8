#pragma once

//! \file fad.hpp
//! \brief Helpers for FAD types

#include "defines.hpp"

namespace calibr8 {

//! \brief Get the value of a scalar type
//! \tparam T the scalar type
//! \param x The input to get the value of
template <typename T>
double val(T const& x);

//! \cond
// template specializations below
template <>
inline double val<double>(double const& x) { return x; }

template <>
inline double val<FADT>(FADT const& x) { return x.val(); }

template <>
inline double val<DFADT>(DFADT const& x) { return x.val(); }
//! \endcond

//! \brief Get the derivative value of a scalar type
//! \tparam T The scalar type
//! \param x The input to get the derivative value of
//! \param i The index of the derivative array to print
template <typename T>
double dx(T const& x, int i);

//! \cond
// template specializations below
template <>
inline double dx<double>(double const&, int) { return 0.; }

template <>
inline double dx<FADT>(FADT const& x, int i) { return x.fastAccessDx(i); }

template <>
inline double dx<DFADT>(DFADT const& x, int i) { return x.fastAccessDx(i); }
//! \endcond

//! \brief Get the number of derivatives of a scalar type
//! \tparam T The scalar type
//! \param x The input to number of derivatives for
template <typename T>
double num_derivs(T const& x);

//! \cond
// template specializations below
template <>
inline double num_derivs<double>(double const&) { return 0.; }

template <>
inline double num_derivs<FADT>(FADT const& x) { return x.size(); }

template <>
inline double num_derivs<DFADT>(DFADT const& x) { return x.size(); }
//! \endcond

}
