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

    //! \brief Create the data needed for verification
    //! \details create the fine primal fields and the branch flags
    void create_verification_data();

    //! \brief Set the nested error contribs to the base mesh
    //! \param E_global The global residual error contribs
    //! \param E_local The local residual error contribs
    void set_error(apf::Field* E_global, apf::Field* E_local);

    //! \brief Set the primal fine fields to the values at the previous step
    //! \param R The global/local residuals defining the problem
    //! \param step The current load/time step
    void initialize_primal_fine(
        RCP<Residuals<double>> R,
        int step);

    //! \brief Get a coarse nodal representation of a fine field
    apf::Field* get_coarse(apf::Field* f);

    //! \brief Get the fine primal fields
    //! \details For VERIFICATION discretizations
    Array1D<Fields>& primal_fine() { return m_primal_fine; }

    //! \brief Get the fine primal fields at a step
    //! \details For VERIFICATION discretizations
    Fields& primal_fine(int step) { return m_primal_fine[step]; }

    //! \brief Get the branch paths
    Array3D<bool>& branch_paths() { return m_branch_paths; }

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

    Array1D<Fields> m_primal_fine;
    Array3D<bool> m_branch_paths; /* [load step, elem_set, elem] */

};

}
