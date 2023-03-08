#pragma once

#include "control.hpp"

#define ALWAYS_ASSERT(cond)                           \
  do {                                                \
    if (! (cond)) {                                   \
      char omsg[2048];                                \
      snprintf(omsg, 2048, "%s failed at %s + %d \n", \
        #cond, __FILE__, __LINE__);                   \
      calibr8::assert_fail(omsg);                     \
    }                                                 \
  } while (0)

#define ALWAYS_ASSERT_VERBOSE(cond, msg)                      \
  do {                                                        \
    if (! (cond)) {                                           \
      char omsg[2048];                                        \
      snprintf(omsg, 2048, "%s failed at %s + %d \n %s \n",   \
        #cond, __FILE__, __LINE__, msg);                      \
      calibr8::assert_fail(omsg);                             \
    }                                                         \
  } while(0)

#ifdef NDEBUG
#define DEBUG_ASSERT(cond)
#define DEBUG_ASSERT_VERBOSE(cond, msg)
#else
#define DEBUG_ASSERT(cond) \
  ALWAYS_ASSERT(cond)
#define DEBUG_ASSERT_VERBOSE(cond, msg) \
  ALWAYS_ASSERT_VERBOSE(cond, msg)
#endif
