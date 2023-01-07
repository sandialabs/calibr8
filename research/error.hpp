#pragma once

#include <fstream>

#include <apf.h>
#include "defines.hpp"

namespace calibr8 {

class Physics;

class Error {
  public:
    virtual apf::Field* compute_error(RCP<Physics> physics) = 0;
    virtual void destroy_intermediate_fields() = 0;
    void write_mesh(RCP<Physics> physics, std::string const& file, int ctr);
    void write_pvd(std::string const& file, int nctr);
    virtual void write_history(std::string const& file, double J_ex=0.) = 0;
};

int get_ndofs(int space, RCP<Physics> physics);
int get_nelems(RCP<Physics> physics);
void write_stream(std::string const& path, std::stringstream const& stream);
apf::Field* interp_error_to_cells(apf::Field* eta);

RCP<Error> create_error(ParameterList const& params);

}
