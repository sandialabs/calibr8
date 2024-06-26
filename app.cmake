capp_app(
  ROOT_PACKAGES
  calibr8
)

find_program(CALIBR8_MPICC NAMES mpicc HINTS ENV MPICC)
find_program(CALIBR8_MPICXX NAMES mpicxx HINTS ENV MPICXX)

set(CALIBR8_ZLIB_PATH "${CAPP_INSTALL_ROOT}/zlib-ng/lib/libz.a")
set(CALIBR8_ZOLTAN_PATH "${CAPP_INSTALL_ROOT}/trilinos/lib/libzoltan.a")
set(CALIBR8_METIS_PATH "${CAPP_INSTALL_ROOT}/parmetis/lib/libmetis.a")
set(CALIBR8_PARMETIS_PATH "${CAPP_INSTALL_ROOT}/parmetis/lib/libparmetis.a")
