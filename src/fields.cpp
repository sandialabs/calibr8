#include <apfMesh.h>
#include <apfShape.h>
#include <spr.h>
#include "arrays.hpp"
#include "fad.hpp"
#include "fields.hpp"
#include "macros.hpp"

namespace calibr8 {

int get_num_eqs(int type, int ndims) {
  int neqs = -1;
  if (type == SCALAR) neqs = 1;
  if (type == VECTOR) neqs = ndims;
  if (type == SYM_TENSOR) neqs = (ndims+1)*(ndims)/2;
  if (type == TENSOR) neqs = ndims * ndims;
  return neqs;
}

Array2D<double> get_nodal_components(
    apf::Field* f,
    apf::MeshElement* me,
    int type,
    int num_nodes) {

  Array2D<double> comps;
  resize(comps, num_nodes, 9);

  int const ndims = apf::getMesh(f)->getDimension();
  int const neqs = get_num_eqs(type, ndims);
  apf::Element* fe = apf::createElement(f, me);
  int const num_apf_nodes = apf::countNodes(fe);
  DEBUG_ASSERT(num_nodes == num_apf_nodes);

  if (type == SCALAR) {
    apf::NewArray<double> apf_comps;
    apf::getScalarNodes(fe, apf_comps);
    for (int n = 0; n < num_nodes; ++n) {
      comps[n][0] = apf_comps[n];
    }
  }

  else if (type == VECTOR) {
    apf::NewArray<apf::Vector3> apf_comps;
    apf::getVectorNodes(fe, apf_comps);
    for (int n = 0; n < num_nodes; ++n) {
      for (int eq = 0; eq < neqs; ++eq) {
        comps[n][eq] = apf_comps[n][eq];
      }
    }
  }

  else {
    fail("unsupported field type: %d", type);
  }

  apf::destroyElement(fe);

  return comps;

}

Array1D<double> get_node_components(
    apf::Field* f,
    apf::MeshEntity* elem,
    int node,
    int type) {

  Array1D<double> vals;
  resize(vals, 9);
  int const ndims = apf::getMesh(f)->getDimension();

  if (type == SCALAR) {
    vals[0] = apf::getScalar(f, elem, node);
  }

  if (type == VECTOR) {
    apf::Vector3 apf_vals;
    apf::getVector(f, elem, node, apf_vals);
    for (int i = 0; i < 3; ++i) {
      vals[i] = apf_vals[i];
    }
  }

  if (type == SYM_TENSOR && ndims == 2) {
    apf::Matrix3x3 apf_vals;
    apf::getMatrix(f, elem, node, apf_vals);
    vals[0] = apf_vals[0][0];
    vals[1] = apf_vals[0][1];
    vals[2] = apf_vals[1][1];
  }

  if (type == SYM_TENSOR && ndims == 3) {
    apf::Matrix3x3 apf_vals;
    apf::getMatrix(f, elem, node, apf_vals);
    vals[0] = apf_vals[0][0];
    vals[1] = apf_vals[0][1];
    vals[2] = apf_vals[0][2];
    vals[3] = apf_vals[1][1];
    vals[4] = apf_vals[1][2];
    vals[5] = apf_vals[2][2];
  }

  if (type == TENSOR && ndims == 2) {
    apf::Matrix3x3 apf_vals;
    apf::getMatrix(f, elem, node, apf_vals);
    vals[0] = apf_vals[0][0];
    vals[1] = apf_vals[0][1];
    vals[2] = apf_vals[1][0];
    vals[3] = apf_vals[1][1];
  }

  if (type == TENSOR && ndims == 3) {
    apf::Matrix3x3 apf_vals;
    apf::getMatrix(f, elem, node, apf_vals);
    vals[0] = apf_vals[0][0];
    vals[1] = apf_vals[0][1];
    vals[2] = apf_vals[0][2];
    vals[3] = apf_vals[1][0];
    vals[4] = apf_vals[1][1];
    vals[5] = apf_vals[1][2];
    vals[6] = apf_vals[1][0];
    vals[7] = apf_vals[1][1];
    vals[8] = apf_vals[1][2];
  }

  return vals;

}

Array1D<double> get_components(
    Array1D<FADT> const& xi,
    int ndims,
    int type) {

  Array1D<double> apf_vals = Array1D<double>(9, 0.);

  if (type == SCALAR) {
    apf_vals[0] = val(xi[0]);
  }

  if (type == VECTOR) {
    for (int dim = 0; dim < ndims; ++dim) {
      apf_vals[dim] = val(xi[dim]);
    }
  }

  if (type == SYM_TENSOR && ndims == 2) {
    apf_vals[0] = val(xi[0]);
    apf_vals[1] = val(xi[1]);
    apf_vals[3] = val(xi[1]);
    apf_vals[4] = val(xi[2]);
  }

  if (type == TENSOR && ndims == 2) {
    apf_vals[0] = val(xi[0]);
    apf_vals[1] = val(xi[1]);
    apf_vals[3] = val(xi[2]);
    apf_vals[4] = val(xi[3]);
  }

  if (type == SYM_TENSOR && ndims == 3) {
    apf_vals[0] = val(xi[0]);
    apf_vals[1] = val(xi[1]);
    apf_vals[2] = val(xi[2]);
    apf_vals[3] = val(xi[1]);
    apf_vals[4] = val(xi[3]);
    apf_vals[5] = val(xi[4]);
    apf_vals[6] = val(xi[2]);
    apf_vals[7] = val(xi[4]);
    apf_vals[8] = val(xi[5]);
  }

  if (type == TENSOR && ndims == 3) {
    for (int comp = 0; comp < 9; ++comp) {
      apf_vals[comp] = val(xi[comp]);
    }
  }

  return apf_vals;
}

Array1D<double> get_components(
    Array1D<double> const& chi,
    int ndims,
    int type) {

  Array1D<double> apf_vals = Array1D<double>(9, 0.);

  if (type == SCALAR) {
    apf_vals[0] = chi[0];
  }

  if (type == VECTOR) {
    for (int dim = 0; dim < ndims; ++dim) {
      apf_vals[dim] = chi[dim];
    }
  }

  if (type == SYM_TENSOR && ndims == 2) {
    apf_vals[0] = chi[0];
    apf_vals[1] = chi[1];
    apf_vals[3] = chi[1];
    apf_vals[4] = chi[2];
  }

  if (type == TENSOR && ndims == 2) {
    apf_vals[0] = chi[0];
    apf_vals[1] = chi[1];
    apf_vals[3] = chi[2];
    apf_vals[4] = chi[3];
  }

  if (type == SYM_TENSOR && ndims == 3) {
    apf_vals[0] = chi[0];
    apf_vals[1] = chi[1];
    apf_vals[2] = chi[2];
    apf_vals[3] = chi[1];
    apf_vals[4] = chi[3];
    apf_vals[5] = chi[4];
    apf_vals[6] = chi[2];
    apf_vals[7] = chi[4];
    apf_vals[8] = chi[5];
  }

  if (type == TENSOR && ndims == 3) {
    for (int comp = 0; comp < 9; ++comp) {
      apf_vals[comp] = chi[comp];
    }
  }

  return apf_vals;
}


static void interpolate_to_from(
    apf::Field* qp,
    apf::Field* nodal) {
  apf::Mesh* m = apf::getMesh(qp);
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m->begin(m->getDimension());
  int const q_order = apf::getShape(qp)->getOrder();
  Array1D<double> field_comps(9);
  while ((elem = m->iterate(elems))) {
    apf::MeshElement* me = apf::createMeshElement(m, elem);
    apf::Element* nodal_elem = apf::createElement(nodal, me);
    int const npts = apf::countIntPoints(me, q_order);
    for (int pt = 0; pt < npts; ++pt) {
      apf::Vector3 iota;
      apf::getIntPoint(me, q_order, pt, iota);
      apf::getComponents(nodal_elem, iota, &(field_comps[0]));
      apf::setComponents(qp, elem, pt, &(field_comps[0]));
    }
  }
  m->end(elems);
}

apf::Field* enrich_nodal_field(
    apf::Field* z_H,
    apf::FieldShape* fine_shape) {
  apf::Mesh* m = apf::getMesh(z_H);
  int const fine_order = fine_shape->getOrder();
  int const vtype = apf::getValueType(z_H);
  std::string name = "enriched_";
  name += apf::getName(z_H);
  apf::Field* qp = apf::createIPField(m, name.c_str(), vtype, fine_order);
  interpolate_to_from(qp, z_H);
  apf::Field* nodal = spr::recoverField(qp);
  apf::destroyField(qp);
  apf::renameField(nodal, name.c_str());
  (void)fine_shape;
  return nodal;
}

apf::Field* enrich_qp_field(
    apf::Field* phi_H,
    apf::FieldShape* fine_shape) {
  apf::Mesh* m = apf::getMesh(phi_H);
  int const vtype = apf::getValueType(phi_H);
  std::string name = "enriched_";
  name += apf::getName(phi_H);
  apf::Field* nodal = spr::recoverField(phi_H);
  apf::Field* qp = apf::createField(m, name.c_str(), vtype, fine_shape);
  interpolate_to_from(qp, nodal);
  apf::destroyField(nodal);
  return qp;
}

apf::Field* subtract(apf::Field* fine, apf::Field* coarse) {
  std::string const bname = apf::getName(fine);
  std::string const name = bname + "_diff";
  apf::Mesh* m = apf::getMesh(fine);
  int const vt = apf::getValueType(fine);
  int const ncomps = apf::countComponents(fine);
  apf::FieldShape* shape = apf::getShape(fine);
  ALWAYS_ASSERT(vt == apf::getValueType(coarse));
  ALWAYS_ASSERT(m == apf::getMesh(coarse));
  ALWAYS_ASSERT(shape == apf::getShape(coarse));
  ALWAYS_ASSERT(ncomps == apf::countComponents(coarse));
  apf::Field* diff = apf::createField(m, name.c_str(), vt, shape);
  double f_comps[3];
  double c_comps[3];
  double diff_comps[3];
  apf::MeshEntity* vtx;
  apf::MeshIterator* it = m->begin(0);
  while ((vtx = m->iterate(it))) {
    apf::getComponents(fine, vtx, 0, &(f_comps[0]));
    apf::getComponents(coarse, vtx, 0, &(c_comps[0]));
    for (int comp = 0; comp < ncomps; ++comp) {
      diff_comps[comp] = f_comps[comp] - c_comps[comp];
    }
    apf::setComponents(diff, vtx, 0, &(diff_comps[0]));
  }
  m->end(it);
  return diff;
}

}
