#include <gtest.h>
#include <control.hpp>

#include <defines.hpp>
  
using namespace calibr8;

TEST(eval, rtc_function) {
  std::string expr = "x + y + z + t";
  double const val1 = eval(expr, 1, 2, 3, 4);
  double const val2 = eval(expr, 1.0, 2.5, 0., 3.2);
  ASSERT_EQ(val1, 10);
  ASSERT_EQ(val2, 6.7);
}

TEST(timer, time) {
  double const t0 = time();
  double const t1 = time();
  double const t = t1 - t0;
  ASSERT_TRUE(t < 10.);
}
