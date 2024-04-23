Compiliation
============

Third-Party Libraries
---------------------

`CALIBR8` has required dependencies on

* `zlib-ng <https://github.com/zlib-ng/zlib-ng>`_ - for file formats
* `hdf5 <https://www.hdfgroup.org/solutions/hdf5/>`_ - for file formats
* `netcdf <https://www.unidata.ucar.edu/software/netcdf/>`_ - for file formats
* `parmetis <https://github.com/KarypisLab/ParMETIS>`_ - for parallel partitioning
* `Trilinos <https://github.com/trilinos/Trilinos>`_ - for automatic differentiation, linear solvers
* `scorec/core <https://github.com/SCOREC/core/tree/master>`_ - for mesh adaptation, finite elements 
* `Eigen <https://gitlab.com/libeigen/eigen>`_ - for small dense linear algebra

zlib-ng
-------

We source `zlib-ng` from the following repo and SHA

.. code-block:: bash
 
  git clone git@github.com:zlib-ng/zlib-ng.git
  e9d0177feafa3adad6ca90cadd44b50389ac4094

.. code-block:: bash

  -DBUILD_SHARED_LIBS=${CALIBR8_BUILD_SHARED_LIBS}
  -DZLIB_COMPAT=ON
