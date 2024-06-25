Theory
======

Governing Equations
-------------------

At its core, CALIBR8 is concerned with solving the balance of linear momentum in the
absence of inertial terms for finite-deformation mechanics in a total-Lagrangian setting.
This can be represented as

.. math::

  \begin{aligned}
  \begin{cases}
  - \nabla \cdot \boldsymbol{P} &= 0, &&\text{in} \quad \mathcal{B}, \\
  \boldsymbol{u} &= \boldsymbol{g}, &&\text{on} \quad \Gamma_g, \\
  \boldsymbol{P} \cdot \boldsymbol{N} &= \boldsymbol{h}, &&\text{on} \quad \Gamma_h.
  \end{cases}
  \end{aligned}

where
:math:`\boldsymbol{P}` denotes the first Piola-Kirchhoff stress tensor
:math:`\mathcal{B}` denotes the domain of interest in the reference configuration,
:math:`\Gamma_g` denotes the portion of the domain boundary on which Dirichlet boundary
conditions :math:`\boldsymbol{u}` are prescribed, and :math:`\Gamma_h` denotes the portion
of the domain boundary on which tractions
:math:`\boldsymbol{P} \cdot \boldsymbol{N}` are prescribed,
