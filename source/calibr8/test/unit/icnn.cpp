#include <ICNN.hpp>

#include <gtest/gtest.h>

using namespace ML;

TEST(input_convex_nn, does_rfad_dfad_work)
{
  DFADT yvar;
  yvar = 1.0;
  yvar.diff(0, 1);
  Sacado::Rad::ADvar<DFADT> y = yvar;
  auto const f = y*y*y;
  Sacado::Rad::ADvar<DFADT>::Gradcomp();
  auto const df = y.adj();
  EXPECT_EQ(df.val(), 3);
  EXPECT_EQ(df.fastAccessDx(0), 6);
  Sacado::Rad::ADcontext<DFADT>::zero_out();
  Sacado::Rad::ADcontext<DFADT>::free_all();
}

#if 0
/* calling rad methods more than once seems to segfault which is crazy
   i.e. I can only call this method and unit tests are fine
   or I can only call the above method and unit tests are fine
   but if I uncomment both, then there is a segfault.
   will need to work out why.
   */
TEST(input_convex_nn, does_eigen_with_rfad_work)
{
  Eigen::Matrix<RAD_DFADT, Eigen::Dynamic, 1> y(1);
  Sacado::Rad::ADvar<DFADT> y_var = DFADT(1, 0, 1.23);
  y(0) = y_var;
  std::cout << y << "\n";
}
#endif

#if 0
/* this segfaults and we need to debug this for a path forward */
TEST(input_convex_nn, scalar)
{
  using NETWORK = FICNN<RAD_DFADT>;
  const char* activation = "softplus";
  std::vector<int> topology = {1,3,1};
  NETWORK nn(activation, topology);
  DFADT y_fad;
  y_fad = 1.0;
  y_fad.diff(0, 1);
  RAD_DFADT y_rad = y_fad;
  NETWORK::Vector y(1);
  y(0) = y_rad;
  nn.evaluate(y);
}
#endif
