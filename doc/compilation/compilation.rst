.. _compilation:

Compiliation
============

CALIBR8 uses the `CApp <https://github.com/sandialabs/capp>`_
build tool, which coordinates the entire build process. In particular,
it acquires and builds all approriate third-party libraries.

Requirements
------------

Install or (on HPC systems) load modules for required tools:

  1. CMake (>3.21)
  2. Git (>2.39)
  3. C and C++ compilers (with C++17 support). If needed, use the
     `CC` and `CXX` environment variables you want to distinguish them
     from others in the `PATH`.
  4. MPI with C and C++ compiler support
  5. Python 3
  6. Ensure that you have SSH keys set up for `GitHub <https://github.com>`_
     on the machine you are using.

Scripts to set up this environment are provided for some common platforms:

============================   ================================================
Command                        Platform
============================   ================================================
`source env/linux-shared.sh`   Generic LINUX environments with shared libraries
`source env/osx-static.sh`     Mac laptop with static libraries
`source env/osx-shared.sh`     Mac laptop with shared libraries
`source env/cee-static.sh`     CEE workstations with static libraries
`source env/cee-shared.sh`     CEE workstations with shared libraries
`source env/toss3-static.sh`   TOSS3 capacity clusters 
============================   ================================================

Choose a flavor
---------------

"Flavors" in CApp control different 'variants' of the build. In many CApp
projects, a flavor corresponds to the underlying HPC architecture and compile
options (e.g. v100 GPUs, A100 APUs, etc..). Presently, we slightly abuse this
notion of a flavor to simply offer users a convenient way to build on specific
machines.

============================   ================================================
Flavor                         Description 
============================   ================================================
`linux-shared`                 LINUX environments with shared libraries
`cee-shared`                   CEE environments with shared libraries
`cee-static`                   CEE environments with static libraries
`osx-static`                   Mac environments with static libraries
`osx-shared`                   Mac environments with shared libraries
`toss3-static`                 TOSS3 capacity clusters
============================   ================================================

If you used one of the environment scripts in the `env/` subdirectory, it also
exports an environment variable `CAPP_FLAVOR` with the most appropriate flavor
for this platform. If you are setting up your own environment, then you can
export this environment variable

.. code-block:: bash

  export CAPP_FLAVOR=linx-shared

Source the CApp setup script
----------------------------

.. code-block:: bash

  source capp-setup.sh

Run the CApp command
--------------------

Once the CApp setup script has been sourced, you can run the CApp command,
optionally specifying the number of cores to compile with (`-j`) and
HTTP proxy if necessary (`--proxy`):

.. code-block:: bash

  capp build -j 4 --proxy http://proxy.sandia.gov:80

Load the CALIBR8 environment
----------------------------

Assuming everything goes well, you can put the `CALIBR8` code  into your
`PATH` and a `python` interpreter that has access to all the relevant
Python modules with the CApp `load` command:

.. code-block:: bash

  capp load

Conversely, you can "tear down" (remove these from your `PATH`) the
enviroment using:

.. code-block:: bash

  capp unload
