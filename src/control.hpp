#pragma once

//! \file control.hpp
//! \brief Methods for code execution control

#include <string>

//! \brief All calibr8 symbols are contained in this namespace
namespace calibr8 {

//! \brief Initialize parallel services (MPI, PCU, Kokkos)
void initialize();

//! \brief Finalize parallel services (MPI, PCU, Kokkos)
void finalize();

//! \brief Print a formatted message on rank 0 to the terminal
//! \param msg The formatted message
void print(const char* msg, ...);

//! \brief Fail the application with a formatted message
//! \param why The formatted failure message
void fail(const char* why, ...) __attribute__((noreturn));

//! \brief The failure mechanism for assert macros
//! \param why The formatted failure message
void assert_fail(const char* why, ...) __attribute__((noreturn));

//! \brief Evaluate a parsed string expression as a function of space/time
//! \param v The string expression in the form "f(x,t)"
//! \param x The x-spatial location
//! \param y The y-spatial location
//! \param z The z-spatial location
//! \param t The evaluation time
double eval(
    std::string const& v,
    double x,
    double y,
    double z,
    double t);

//! \brief Get the current CPU time for timers
double time();

}
