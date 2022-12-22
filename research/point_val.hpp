#include "control.hpp"
#include "disc.hpp"
#include "linalg.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class PointVal : public QoI<T> {

  private:

    int m_eq = 0;
    std::string m_node_set;

  public:

    PointVal(ParameterList const& params) : QoI<T>() {
      m_node_set = params.get<std::string>("node");
      m_eq = params.get<int>("eq");
    }

    ~PointVal() override {
    }

    void at_point(
        apf::Vector3 const&,
        double,
        double,
        RCP<Residual<T>>,
        RCP<Disc> disc) override {
    }

    void post(int space, RCP<Disc> disc, RCP<VectorT> U, System* sys) override {
      this->m_value = 0.;
      auto nodes = disc->nodes(space, m_node_set);
      apf::MeshEntity* vtx = 0;
      if (nodes.size() > 0) {
        ASSERT(nodes.size() == 1);
        vtx = nodes[0].entity;
      }
      if (vtx && disc->apf_mesh()->isOwned(vtx)) {
        LO row = disc->get_lid(space, vtx, 0, m_eq);
        auto U_data = U->get1dView();
        this->m_value = U_data[row];
        if (sys) {
          auto dJdU_data = sys->b->get1dViewNonConst();
          dJdU_data[row] = 1.;
        }
      }
    }

};

}
