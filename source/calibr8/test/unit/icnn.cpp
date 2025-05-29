#include <ICNN.hpp>

#include <gtest/gtest.h>

using namespace ML;

/* Sacado::Rad tries to be 'too smart' at re-using memory
   so we yeet memory cleaning and reinit'ing calls everywhere
   we can */

TEST(input_convex_nn, does_rfad_dfad_work)
{
  Sacado::Rad::ADcontext<DFADT>::re_init();
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

TEST(input_convex_nn, does_rfad_dfad_work2)
{
  Sacado::Rad::ADcontext<DFADT>::re_init();
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

TEST(input_convex_nn, does_eigen_with_rfad_work)
{
  Sacado::Rad::ADcontext<DFADT>::re_init();
  Eigen::Matrix<RAD_DFADT, Eigen::Dynamic, 1> y(1);
  Sacado::Rad::ADvar<DFADT> y_var = DFADT(1, 0, 1.23);
  y(0) = y_var;
  std::cout << y << "\n";
  Sacado::Rad::ADcontext<DFADT>::zero_out();
  Sacado::Rad::ADcontext<DFADT>::free_all();
}

TEST(input_convex_nn, scalar)
{
  using NETWORK = FICNN<double>;
  const char* activation = "softplus";
  std::vector<int> topology = {1,3,1};
  NETWORK nn(activation, topology);
  NETWORK::Vector y(1);
  y(0) = 2.0;
  nn.evaluate(y);
}

#if 0
/* this still segfaults */
TEST(input_convex_nn, rfad_dfadt)
{
  Sacado::Rad::ADcontext<DFADT>::re_init();
  using NETWORK = FICNN<RAD_DFADT>;
  const char* activation = "softplus";
  std::vector<int> topology = {1,3,1};
  NETWORK nn(activation, topology);
  DFADT y_fad;
  y_fad = 1.0;
  y_fad.diff(0, 1);
  RAD_DFADT y_rad = y_fad;
  NETWORK::Vector y(1);
  y(0) = RAD_DFADT(y_fad);
  nn.evaluate(y);
}
#endif
