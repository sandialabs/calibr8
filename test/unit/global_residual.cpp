#include <gtest/gtest.h>
#include <global_residual.hpp>

#include <mechanics.hpp>

using namespace calibr8;

TEST(global_residual, factory_creation) {
  ParameterList params;
  params.set<std::string>("type", "mechanics");
  RCP<GlobalResidual<double>> R = create_global_residual<double>(params, 3);
  EXPECT_EQ(R->num_residuals(), 2);
  EXPECT_EQ(R->num_eqs(0), 3);
  EXPECT_EQ(R->num_eqs(1), 1);
}
