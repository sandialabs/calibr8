#include <apf.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include "arrays.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "macros.hpp"
#include "tbcs.hpp"

namespace calibr8 {

using Teuchos::Array;
using Teuchos::getValue;

static void apply_primal_tbc(
    Array<std::string> const& a,
    RCP<Disc> disc,
    Array1D<RCP<VectorT>>& R,
    double t) {

  // variables to set BC
  apf::Vector3 x(0, 0, 0);
  apf::Vector3 iota(0, 0, 0);
  apf::Vector3 T(0, 0, 0);
  apf::NewArray<double> BF;

  // information about the BC from the input
  int const i = std::stoi(a[0]); // the residual index
  std::string const set = a[1];  // the side set to apply the bc to
  Array1D<std::string> vals;
  for (int d = 1; d <= disc->num_dims(); ++d) {
    vals.push_back(a[1 + d]);
  }

  // mesh specific information
  int const num_dims = disc->num_dims();
  apf::Mesh* mesh = disc->apf_mesh();
  apf::FieldShape* gv_shape = disc->gv_shape();
  apf::FieldShape* lv_shape = disc->lv_shape();
  int const q_order = lv_shape->getOrder();
  SideSet const& sides = disc->sides(set);
  Teuchos::ArrayRCP<double> R_data = R[i]->get1dViewNonConst();

  // loop over all faces for this BC
  for (size_t s = 0; s < sides.size(); ++s) {

    // get some info about the basis for the face
    apf::MeshEntity* side = sides[s];
    apf::MeshElement* me = apf::createMeshElement(mesh, side);
    apf::EntityShape* es = gv_shape->getEntityShape(mesh->getType(side));
    int const num_nodes = es->countNodes();

    // loop over integration points along the face
    int const npts = apf::countIntPoints(me, q_order);
    for (int pt = 0; pt < npts; ++pt) {

      // get info at the current integration point
      apf::getIntPoint(me, q_order, pt, iota);
      double const w = apf::getIntWeight(me, q_order, pt);
      double const dv = apf::getDV(me, iota);
      apf::mapLocalToGlobal(me, iota, x);
      apf::getBF(gv_shape, me, iota, BF);

      // evaluate the traction
      for (int d = 0; d < num_dims; ++d) {
        T[d] = eval(vals[d], x[0], x[1], x[2], t);
      }

      // modify the residual for the applied traction
      for (int n = 0; n < num_nodes; ++n) {
        for (int d = 0; d < num_dims; ++d) {
          LO row = disc->get_lid(side, i, n, d);
          R_data[row] -= T[d] * BF[n] * w * dv;
        }
      }

    }

    // clean up memory
    apf::destroyMeshElement(me);

  }

}

void apply_primal_tbcs(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<RCP<VectorT>>& R,
    double t) {
  for (auto it = tbcs.begin(); it != tbcs.end(); ++it) {
    auto entry = tbcs.entry(it);
    auto a = getValue<Array<std::string>>(entry);
    apply_primal_tbc(a, disc, R, t);
  }
}

void eval_tbc_error_contributions(
    Array<std::string> const& a,
    RCP<Disc> disc,
    Array1D<apf::Field*> z,
    apf::Field* R_error,
    double t) {

  // variables to set BC
  apf::Vector3 x(0, 0, 0);
  apf::Vector3 iota(0, 0, 0);
  apf::Vector3 T(0, 0, 0);
  apf::NewArray<double> BF;

  // information about the BC from the input
  int const i = std::stoi(a[0]); // the residual index
  std::string const set = a[1];  // the side set to apply the bc to
  Array1D<std::string> vals;
  for (int d = 1; d <= disc->num_dims(); ++d) {
    vals.push_back(a[1 + d]);
  }

  apf::Up elems;
  apf::Field* z_field = z[i];

  // mesh specific information
  int const num_dims = disc->num_dims();
  apf::Mesh* mesh = disc->apf_mesh();
  apf::FieldShape* gv_shape = disc->gv_shape();
  apf::FieldShape* lv_shape = disc->lv_shape();
  int const q_order = lv_shape->getOrder();
  SideSet const& sides = disc->sides(set);

  // loop over all faces for this BC
  for (size_t s = 0; s < sides.size(); ++s) {

    // get some info about the basis for the face
    apf::MeshEntity* side = sides[s];
    apf::MeshElement* me = apf::createMeshElement(mesh, side);
    apf::Element* z_elem = apf::createElement(z_field, me);
    apf::EntityShape* es = gv_shape->getEntityShape(mesh->getType(side));
    int const num_nodes = es->countNodes();

    // get the adjoint solution at side nodes
    apf::NewArray<apf::Vector3> z_values;
    apf::getVectorNodes(z_elem, z_values);

    // loop over integration points along the face
    int const npts = apf::countIntPoints(me, q_order);
    for (int pt = 0; pt < npts; ++pt) {

      // get info at the current integration point
      apf::getIntPoint(me, q_order, pt, iota);
      double const w = apf::getIntWeight(me, q_order, pt);
      double const dv = apf::getDV(me, iota);
      apf::mapLocalToGlobal(me, iota, x);
      apf::getBF(gv_shape, me, iota, BF);

      // evaluate the traction
      for (int d = 0; d < num_dims; ++d) {
        T[d] = eval(vals[d], x[0], x[1], x[2], t);
      }

      // compute the error contribution
      double E_side = 0.;
      for (int n = 0; n < num_nodes; ++n) {
        for (int d = 0; d < num_dims; ++d) {
          E_side -= T[d] * BF[n] * w * dv * z_values[n][d];
        }
      }

      // contribute the error to the error field
      mesh->getUp(side, elems);
      ALWAYS_ASSERT(elems.n == 1);
      apf::MeshEntity* elem = elems.e[0];
      double E_R = apf::getScalar(R_error, elem, 0);
      apf::setScalar(R_error, elem, 0, E_R + E_side);

    }

    // clean up
    apf::destroyMeshElement(me);

  }

}

double sum_tbc_error_contributions(
    Array<std::string> const& a,
    RCP<Disc> disc,
    Array1D<apf::Field*> z,
    double t) {

  double e = 0.;

  // variables to set BC
  apf::Vector3 x(0, 0, 0);
  apf::Vector3 iota(0, 0, 0);
  apf::Vector3 T(0, 0, 0);
  apf::NewArray<double> BF;

  // information about the BC from the input
  int const i = std::stoi(a[0]); // the residual index
  std::string const set = a[1];  // the side set to apply the bc to
  Array1D<std::string> vals;
  for (int d = 1; d <= disc->num_dims(); ++d) {
    vals.push_back(a[1 + d]);
  }

  apf::Up elems;
  apf::Field* z_field = z[i];

  // mesh specific information
  int const num_dims = disc->num_dims();
  apf::Mesh* mesh = disc->apf_mesh();
  apf::FieldShape* gv_shape = disc->gv_shape();
  apf::FieldShape* lv_shape = disc->lv_shape();
  int const q_order = lv_shape->getOrder();
  SideSet const& sides = disc->sides(set);

  // loop over all faces for this BC
  for (size_t s = 0; s < sides.size(); ++s) {

    // get some info about the basis for the face
    apf::MeshEntity* side = sides[s];
    apf::MeshElement* me = apf::createMeshElement(mesh, side);
    apf::Element* z_elem = apf::createElement(z_field, me);
    apf::EntityShape* es = gv_shape->getEntityShape(mesh->getType(side));
    int const num_nodes = es->countNodes();

    // get the adjoint solution at side nodes
    apf::NewArray<apf::Vector3> z_values;
    apf::getVectorNodes(z_elem, z_values);

    // loop over integration points along the face
    int const npts = apf::countIntPoints(me, q_order);
    for (int pt = 0; pt < npts; ++pt) {

      // get info at the current integration point
      apf::getIntPoint(me, q_order, pt, iota);
      double const w = apf::getIntWeight(me, q_order, pt);
      double const dv = apf::getDV(me, iota);
      apf::mapLocalToGlobal(me, iota, x);
      apf::getBF(gv_shape, me, iota, BF);

      // evaluate the traction
      for (int d = 0; d < num_dims; ++d) {
        T[d] = eval(vals[d], x[0], x[1], x[2], t);
      }

      // compute the error contribution
      double E_side = 0.;
      for (int n = 0; n < num_nodes; ++n) {
        for (int d = 0; d < num_dims; ++d) {
          E_side += T[d] * BF[n] * w * dv * z_values[n][d];
        }
      }

      // contribute the error to the error field
      e += E_side;

    }

    // clean up
    apf::destroyMeshElement(me);

  }

  return e;

}

void eval_tbcs_error_contributions(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<apf::Field*> zfields,
    apf::Field* R_error,
    double t) {
  for (auto it = tbcs.begin(); it != tbcs.end(); ++it) {
    auto entry = tbcs.entry(it);
    auto a = getValue<Array<std::string>>(entry);
    eval_tbc_error_contributions(a, disc, zfields, R_error, t);
  }
}

double sum_tbcs_error_contributions(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<apf::Field*> zfields,
    double t) {
  double tbcs_error = 0.;
  for (auto it = tbcs.begin(); it != tbcs.end(); ++it) {
    auto entry = tbcs.entry(it);
    auto a = getValue<Array<std::string>>(entry);
    tbcs_error += sum_tbc_error_contributions(a, disc, zfields, t);
  }
  return tbcs_error;
}

}

