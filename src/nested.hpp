#pragma once

//! \file nested.hpp
//! \brief A nested discretization container for uniformly refined meshes

#include "disc.hpp"

namespace calibr8 {

using EntArray = Array1D<apf::MeshEntity*>;

class NestedDisc : public Disc {
  public:

    //! \brief Create the nested discretization
    //! \param disc The original coarse discretization type
    //! \param disc_type The disc type (NESTED/VERIFICATION)
    NestedDisc(RCP<Disc> disc, int disc_type = NESTED);

    //! \brief Destroy the nested discretization object
    ~NestedDisc();

    //! \brief Set the nested error contribs to the base mesh
    //! \param E_global The global residual error contribs
    //! \param E_local The local residual error contribs
    void set_error(apf::Field* E_global, apf::Field* E_local);

    //! \brief Get a coarse nodal representation of a fine field
    apf::Field* get_coarse(apf::Field* f);

  private:

    void number_elems();
    void copy_mesh();
    void tag_old_verts();
    void refine();
    void store_old_verts();
    void create_primal(RCP<Disc> disc);

  private:

    apf::Mesh2* m_base_mesh = nullptr;
    apf::MeshTag* m_old_vtx_tag = nullptr;
    apf::MeshTag* m_new_vtx_tag = nullptr;
    EntArray m_base_elems;
    EntArray m_old_vertices;


};

}
