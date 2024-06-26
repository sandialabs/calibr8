#include <gtest.h>
#include <control.hpp>

int main(int argc, char** argv) {
  calibr8::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  int const result = RUN_ALL_TESTS();
  calibr8::finalize();
  return result;
}
