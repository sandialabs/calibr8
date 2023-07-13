#include <PCU.h>
#include "disc.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "avg_element_stress.hpp"

namespace calibr8 {

template <typename T>
AvgElementStress<T>::AvgElementStress(ParameterList const& params) {
  m_elem_set = params.get<std::string>("elem set");
  m_stress_idx = params.get<Teuchos::Array<int>>("stress component").toVector();
  if (params.isParameter("transform to polar")) {
    m_transform_to_polar = params.get<bool>("transform to polar");
  }
  if (params.isParameter("step")) {
    m_qoi_eval_step = params.get<int>("step");
  }
}

template <typename T>
AvgElementStress<T>::~AvgElementStress() {
}

template <typename T>
void AvgElementStress<T>::before_elems(RCP<Disc> disc, int step) {
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;
  m_elem_set_idx = disc->elem_set_idx(m_elem_set);
  m_volume = 0.;
  ElemSet const& elems = disc->elems(m_elem_set);
  for (auto elem : elems) {
    apf::MeshElement* me = apf::createMeshElement(this->m_mesh, elem);
    apf::Vector3 iota;
    apf::getIntPoint(me, 1, 0, iota);
    double const w = apf::getIntWeight(me, 1, 0);
    double const dv = apf::getDV(me, iota);
    m_volume += w * dv;
    apf::destroyMeshElement(me);
  }
  m_volume = PCU_Add_Double(m_volume);
  print(" > qoi volume: %.15e", m_volume);
}

template <typename T>
void AvgElementStress<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  this->initialize_value_pt();

  if (elem_set != m_elem_set_idx) return;

  if (this->m_step == m_qoi_eval_step) {

    Tensor<T> sigma = local->cauchy(global);

    if (m_transform_to_polar) {
      apf::Vector3 x(0., 0., 0.);
      apf::mapLocalToGlobal(this->m_mesh_elem, iota, x);
      double const theta = std::atan2(x[1], x[0]);
      Tensor<double> Q = minitensor::zero<double>(3);
      Q(0, 0) = std::cos(theta);
      Q(0, 1) = std::sin(theta);
      Q(1, 0) = -Q(0, 1);
      Q(1, 1) = Q(0, 0);
      Q(2, 2) = 1.;
      sigma = Q * sigma * minitensor::transpose(Q);
    }

    int const i_idx = m_stress_idx[0];
    int const j_idx = m_stress_idx[1];
    this->value_pt += sigma(i_idx, j_idx) * w * dv / m_volume;
  }

}

template class AvgElementStress<double>;
template class AvgElementStress<FADT>;

}
