capp_app(
  ROOT_PACKAGES
  eigen
  gmodel
  trilinos
)

find_program(CALIBR8_MPICC NAMES mpicc HINTS ENV MPICC)
find_program(CALIBR8_MPICXX NAMES mpicxx HINTS ENV MPICXX)
