.. _forward:

=========================
Running a forward problem
=========================

Once the CALIBR8 project has been compiled and the environment
has been loaded with the command `capp load` (see the instructions
:ref:`compilation`.), running a forward problem is as simple as
calling the CALIBR8 executable `primal`, which should now exist in
your `PATH` with an input YAML file.

.. code-block:: bash

  primal <input.yaml>

Example Input YAML FILE
=======================

Below are the contents of an example input YAML file, that we will
go through line by line in subsequent sections:

.. code-block:: yaml

  example_input:

    problem:
      name: J2_finite_2D

    discretization:
      geom file: 'notch2D.dmg'
      mesh file: 'notch2D.smb'
      assoc file: 'notch2D.txt'
      num steps: 8
      step size: 1.

    residuals:
      global residual:
        type: 'mechanics'
        nonlinear max iters: 15
        nonlinear absolute tol: 1.e-12
        nonlinear relative tol: 1.e-12
        print convergence: true
      local residual:
        type: 'hyper_J2_plane_strain'
        nonlinear max iters: 500
        nonlinear absolute tol: 1.e-12
        nonlinear relative tol: 1.e-12
        materials:
          body:
            E: 1000.
            nu: 0.25
            K: 100.
            Y: 10.
            Y_inf: 0.
            delta: 0.

    # bc name: [resid_idx, eq, node_set_name, value]
    dirichlet bcs:
      expression:
        bc 1: [0, 0, xmin, 0.0]
        bc 2: [0, 1, ymin, 0.0]
        bc 3: [0, 1, ymax, 0.001 * t]

    linear algebra:
      Linear Solver Type: "Belos"
      Preconditioner Type: "Teko"
      Linear Solver Types:
        Belos:
          Solver Type: "Block GMRES"
          Solver Types:
            Block GMRES:
              Convergence Tolerance: 1.e-12
              Output Frequency: 10
              Output Style: 1
              Verbosity: 33
              Maximum Iterations: 200
              Block Size: 1
              Num Blocks: 200
              Flexibile Gmres: false
          VerboseObject:
            Output File: "none"
            Verbosity Level: "none"
      Preconditioner Types:
        Teko:
          Inverse Type: "BGS2x2"
          Write Block Operator: false
          Test Block Operator: false
          Inverse Factory Library:
            BGS2x2:
              Type: "Block Gauss-Seidel"
              Use Upper Triangle: false
              Inverse Type 1: "AMG2"
              Inverse Type 2: "AMG1"
            AMG2:
              Type: "MueLu"
              number of equations: 2
              verbosity: "none"
              'problem: symmetric': false
            AMG1:
              Type: "MueLu"
              verbosity: "none"
              number of equations: 1
              'problem: symmetric': false
            GS:
              Type: "Ifpack2"
              Overlap: 1
              Ifpack2 Settings:
                'relaxation: type': "Gauss-Seidel"

Problem
-------

The problem block will specify various aspects of the problem that
you are about to run. In this case, we are only specifying a variable
called `name`, which will uniquely name the primal run. All output
from this run will be output to the directory `name`.

.. code-block:: yaml

    problem:
      name: J2_finite_2D


Discretization
--------------

The discretization block will specify all aspects of an individual
problem's discretization. For all problems, we require a geometric
model, a mesh of that geometric model, and an associations file
linking together collections of geometric objects to a unique string.
The generation and usage of these files is nontrivial and is
covered in depth in a different section (TODO PUT A REFERENCE HERE
ONCE THIS IS DOCUMENTED).

The number of steps and the step size determines how the problem
is loaded. In this example, we take 8 load steps with a psuedo-time
increment of :math:`\Delta t = 1.0`. Later on, in the boundary
conditions block, this pseudo-time :math:`t` will be used to specify
boundary conditions at load increments. (THIS SYNTAX MAY CHANGE AS
WE MOVE TO CONSIDER TRUE TRANSIENT PHYSICS AND NOT QUASI-STATIC
BEHAVIOR).

Additionally, there are some pre-included geometries and meshes
for use in testing CALIBR8 available in the
`repository <https://github.com/sandialabs/calibr8/tree/main/source/calibr8/test/mesh>`_.

.. code-block:: yaml

    discretization:
      geom file: 'notch2D.dmg'
      mesh file: 'notch2D.smb'
      assoc file: 'notch2D.txt'
      num steps: 8
      step size: 1.

Residuals
---------

In general, CALIBR8 considers the coupling of two residual systems:
a `global` residual and a `local` residual. In this case, the global
residual corresponds to the overall governing kinematic equations
(the balance of linear momentum in the absence of inertial terms)
and the local residual corresponds to the constitutive model equations
solved at each integration point inside of an element. Here, we
have specified some parameters that are largely self explanatory.

Of note, perhaps, is that we have chosen a 2D plane strain finite
deformation plasticity model. For this model, we have chosen
some benign material parameters for the elastic modulus :math:`E`,
Poisson's ratio :math:`\nu`, the yield strength :math:`Y`, the
linear hardening modulus :math:`K`, and have set other parameters
to :math:`0`.

.. code-block:: yaml

    residuals:
      global residual:
        type: 'mechanics'
        nonlinear max iters: 15
        nonlinear absolute tol: 1.e-12
        nonlinear relative tol: 1.e-12
        print convergence: true
      local residual:
        type: 'hyper_J2_plane_strain'
        nonlinear max iters: 500
        nonlinear absolute tol: 1.e-12
        nonlinear relative tol: 1.e-12
        materials:
          body:
            E: 1000.
            nu: 0.25
            K: 100.
            Y: 10.
            Y_inf: 0.
            delta: 0.

Boundary Conditions
-------------------

In the dirichlet bcs block, we set up some simple boundary conditions
to constrain rigid body rotations/translations, and pull our geometry
at an increment of :math:`0.01*t` on the top `y` face of the notch
specimen domain.

Note that, as the comment suggests, the DBC input structure has
the form: [residual index, equation index, node set name, value as a string].
The node set name comes from the associations file `assoc.txt` set in
the discretization block, and the value is a run-time-compiled string
expression that can be fairly sophisticated.

.. code-block:: yaml

    # bc name: [resid_idx, eq, node_set_name, value]
    dirichlet bcs:
      expression:
        bc 1: [0, 0, xmin, 0.0]
        bc 2: [0, 1, ymin, 0.0]
        bc 3: [0, 1, ymax, 0.001 * t]

Linear Algebra
--------------

The linear algebra block should remain largely untouched unless
you are an experienced user. Of particular note, however, is that the
block below should change from `2` to `3` when switching from a 2D to
a 3D problem.

.. code-block:: yaml

            AMG2:
              Type: "MueLu"
              number of equations: 2
              verbosity: "none"
              'problem: symmetric': false
