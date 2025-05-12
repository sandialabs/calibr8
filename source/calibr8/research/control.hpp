#pragma once

#include <string>

namespace calibr8 {

void initialize();
void finalize();
void print(const char* msg, ...);
void fail(const char* why, ...) __attribute__((noreturn));
void assert_fail(const char* why, ...) __attribute__((noreturn));

double eval(
    std::string const& v,
    double x,
    double y,
    double z,
    double t);

double time();

}

#define ASSERT(cond)                                  \
  do {                                                \
    if (! (cond)) {                                   \
      char omsg[2048];                                \
      snprintf(omsg, 2048, "%s failed at %s + %d \n", \
        #cond, __FILE__, __LINE__);                   \
      calibr8::assert_fail(omsg);                     \
    }                                                 \
  } while (0)
