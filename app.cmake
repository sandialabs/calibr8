capp_app(
  ROOT_PACKAGES
  calibr8
)

find_program(CALIBR8_MPICC NAMES mpicc HINTS ENV MPICC)
find_program(CALIBR8_MPICXX NAMES mpicxx HINTS ENV MPICXX)
