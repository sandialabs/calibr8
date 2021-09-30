#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "surface_mismatch.hpp"

namespace calibr8 {

template <typename T>
SurfaceMismatch<T>::SurfaceMismatch(ParameterList const& params) {
  m_side_set = params.get<std::string>("side set");
}

template <typename T>
SurfaceMismatch<T>::~SurfaceMismatch() {
}

template <typename T>
void SurfaceMismatch<T>::before_elems(RCP<Disc> disc) {

  // set the discretization-based information
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();

  // initialize the surface mesh information
  if (!is_initd) {
    int ndims = this->m_num_dims;
    apf::Mesh* mesh = disc->apf_mesh();
    apf::Downward downward_faces;
    m_mapping.resize(disc->num_elem_sets());
    SideSet const& sides = disc->sides(m_side_set);
    for (int es = 0; es < disc->num_elem_sets(); ++es) {
      std::string const& es_name = disc->elem_set_name(es);
      ElemSet const& elems = disc->elems(es_name);
      m_mapping[es].resize(elems.size());
      for (size_t elem = 0; elem < elems.size(); ++elem) {
        m_mapping[es][elem] = -1;
        apf::MeshEntity* elem_entity = elems[elem];
        int ndown = mesh->getDownward(elem_entity, ndims - 1, downward_faces);
        for (int down = 0; down < ndown; ++down) {
          apf::MeshEntity* downward_entity = downward_faces[down];
          for (apf::MeshEntity* side : sides) {
            if (side == downward_entity) {
              m_mapping[es][elem] = down;
            }
          }
        }
      }
    }
    is_initd = true;
  }

}

template <typename T>
void SurfaceMismatch<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double,
    double) {
  (void)elem_set;
  (void)elem;
  this->value_pt = 0. * global->vector_x(0)[0] + 0. * local->first_value();
}

template class SurfaceMismatch<double>;
template class SurfaceMismatch<FADT>;

}
