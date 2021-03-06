#include "gmodel.hpp"

int main()
{
  using namespace gmod;
  default_size = 0.1;
  auto p0 = new_point3(Vector{0, 0, 0}, 0.05);
  auto p1 = new_point3(Vector{0.2, 0, 0}, 0.05);
  auto p2 = new_point2(Vector{1, 0, 0});
  auto p3 = new_point2(Vector{1, 1, 0});
  auto p4 = new_point2(Vector{0, 1, 0});
  auto p5 = new_point3(Vector{0, 0.2, 0}, 0.05);
  auto l0 = new_line2(p1, p2);
  auto l1 = new_line2(p2, p3);
  auto l2 = new_line2(p3, p4);
  auto l3 = new_line2(p4, p5);
  auto l4 = new_arc2(p5, p0, p1);
  auto loop = new_loop();
  add_use(loop, FORWARD, l0);
  add_use(loop, FORWARD, l1);
  add_use(loop, FORWARD, l2);
  add_use(loop, FORWARD, l3);
  add_use(loop, FORWARD, l4);
  auto f = new_plane2(loop);
  auto b = extrude_face(f, Vector{0, 0, 0.1}).middle;
  write_closure_to_geo(b, "notch.geo");
  write_closure_to_dmg(b, "notch.dmg");
}
