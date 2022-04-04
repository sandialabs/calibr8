#include "avg_disp.hpp"
#include "avg_stress.hpp"
#include "calibration.hpp"
#include "disc.hpp"
#include "fad.hpp"
#include "load_mismatch.hpp"
#include "qoi.hpp"
#include "surface_mismatch.hpp"

namespace calibr8 {

template <typename T>
QoI<T>::QoI() {
}

template <typename T>
QoI<T>::~QoI() {
}

template <typename T>
void QoI<T>::before_elems(RCP<Disc> disc, int step) {

  // set discretization-based information
  m_mesh = disc->apf_mesh();
  m_num_dims = disc->num_dims();
  m_shape = disc->gv_shape();
  m_step = step;

}

template <typename T>
void QoI<T>::set_elem(apf::MeshElement* mesh_elem) {
  m_mesh_elem = mesh_elem;
}

template <typename T>
void QoI<T>::scatter(double& J) {
  J += val(value_pt);
}

template <typename T>
bool QoI<T>::setup_mapping(std::string const& side_set,
    RCP<Disc> disc,
    Array2D<int>& mapping) {

  int ndims = m_num_dims;
  apf::Mesh* m_mesh = disc->apf_mesh();
  apf::Downward downward_faces;
  mapping.resize(disc->num_elem_sets());
  SideSet const& sides = disc->sides(side_set);
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    mapping[es].resize(elems.size());
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      mapping[es][elem] = -1;
      apf::MeshEntity* elem_entity = elems[elem];
      int ndown = m_mesh->getDownward(elem_entity, ndims - 1, downward_faces);
      for (int down = 0; down < ndown; ++down) {
        apf::MeshEntity* downward_entity = downward_faces[down];
        for (apf::MeshEntity* side : sides) {
          if (side == downward_entity) {
            mapping[es][elem] = down;
          }
        }
      }
    }
  }
  return true;
}

template <typename T>
void QoI<T>::preprocess(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input,
    double w,
    double dv) {}

template <typename T>
void QoI<T>::preprocess_finalize(int step) {}

template <typename T>
void QoI<T>::postprocess(double& J) {}

template <>
EVector QoI<double>::eigen_dvector(int) const {
  EVector empty;
  return empty;
}

template <>
EVector QoI<FADT>::eigen_dvector(int nderivs) const {
  EVector dJ(nderivs);
  dJ.setZero();
  for (int i = 0; i < nderivs; ++i) {
    dJ[i] = value_pt.fastAccessDx(i);
  }
  return dJ;
}

template <typename T>
void QoI<T>::unset_elem() {
  m_mesh_elem = nullptr;
}

template <typename T>
void QoI<T>::after_elems() {
  m_num_dims = -1;
  m_step = -1;
  m_mesh = nullptr;
  m_shape = nullptr;
}

template <>
void QoI<double>::initialize_value_pt() {
  value_pt = 0.;
}

template <>
void QoI<FADT>::initialize_value_pt() {
  value_pt = 0.;
  for (int idx = 0.; idx < nmax_derivs; ++idx) {
    value_pt.fastAccessDx(idx) = 0.;
  }
}

template <typename T>
RCP<QoI<T>> create_qoi(ParameterList const& params) {
  std::string const type = params.get<std::string>("type");
  if (type == "average displacement") {
    return rcp(new AvgDisp<T>());
  } else if (type == "average stress") {
    return rcp(new AvgStress<T>(params));
  } else if (type == "surface mismatch") {
    return rcp(new SurfaceMismatch<T>(params));
  } else if (type == "load mismatch") {
    return rcp(new LoadMismatch<T>(params));
  } else if (type == "calibration") {
    return rcp(new Calibration<T>(params));
  } else {
    return Teuchos::null;
  }
}

template class QoI<double>;
template class QoI<FADT>;

template RCP<QoI<double>>
create_qoi<double>(ParameterList const&);

template RCP<QoI<FADT>>
create_qoi<FADT>(ParameterList const&);

}
