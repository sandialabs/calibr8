Compiliation
============

CALIBR8 uses the `CApp<https://github.com/sandialabs/capp>`_
build tool, which coordinates the entire build process. In particular,
it acquires and builds all approriate third-party libraries.

Requirements
------------

1. Install or (on HPC systems) load modules for required tools
    1. CMake (>3.21)
    1. Git (>2.39)
    1. C and C++ compilers (with C++17 support). If needed, use the
       `CC` and `CXX` environment variables you want to distinguish them
       from others in the `PATH`.
    1. MPI with C and C++ compiler support
    1. Python 3
