set(CALIBR8_BUILD_SHARED_LIBS OFF)
set(TRILINOS_EXTRA_LINK_FLAGS "-DTrilinos_EXTRA_LINK_FLAGS=dl")
set(CALIBR8_ZLIB_PATH "${CAPP_INSTALL_ROOT}/zlib-ng/lib/libz.a")
set(CALIBR8_ZOLTAN_PATH "${CAPP_INSTALL_ROOT}/trilinos/lib64/libzoltan.a")
set(CALIBR8_METIS_PATH "${CAPP_INSTALL_ROOT}/parmetis/lib/libmetis.a")
set(CALIBR8_PARMETIS_PATH "${CAPP_INSTALL_ROOT}/parmetis/lib/libparmetis.a")