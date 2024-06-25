#include "gmodel.hpp"

using namespace gmod;

int main() {
  default_size = 1.0;
  auto cube = new_cube({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1});
  write_closure_to_geo(cube, "cube.geo");
  write_closure_to_dmg(cube, "cube.dmg");
}
