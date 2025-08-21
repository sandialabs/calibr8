#include "control.hpp"
#include "snap.hpp"

namespace calibr8 {

using Teuchos::Array;
using Teuchos::ParameterList;

apf::Vector3 closest_point(
    apf::Vector3 const& x,
    apf::Vector3 const& c,
    double const r)
{
  apf::Vector3 xmc = x - c;
  xmc[2] = 0.0; /* shenanigans for axis-aligned assumption */
  apf::Vector3 const xmcn = xmc.normalize();
  apf::Vector3 p = xmcn * r;
  p[2] = x[2]; /* same here */
  return p;
}

void snap_nodes(
    RCP<Disc> disc,
    ParameterList const& params)
{
  if (!params.isSublist("snapping")) return;
  print("SNAPPING NODES\n");
  ParameterList const& snap_params = params.sublist("snapping");
  std::string const side_set_name = snap_params.get<std::string>("side set");
  SideSet const& sides = disc->sides(side_set_name);
  /* we are snapping only to cylindrical analytic geoms right now
     we assume this cylinder is axis-aligned in the z-direction */
  auto const center = snap_params.get<Array<double>>("center");
  double const radius = snap_params.get<double>("radius");
  apf::Vector3 c(center[0], center[1], center[2]);
  apf::Downward verts;
  apf::Mesh2* m = disc->apf_mesh();
  for (apf::MeshEntity* side : sides) {
    int const nverts = m->getDownward(side, 0, verts);
    for (int v = 0; v < nverts; ++v) {
      apf::Vector3 x;
      m->getPoint(verts[v], 0, x);
      apf::Vector3 xmc = x - c;
      double const r = std::sqrt(xmc[0]*xmc[0] + xmc[1]*xmc[1]);
      if (std::abs(radius - r) > 1.e-8) {
        apf::Vector3 const p = closest_point(x, c, radius);
        m->setPoint(verts[v], 0, p);
      }
    }
  }
}

}
