capp_package(
  GIT_URL https://github.com/zlib-ng/zlib-ng.git
  COMMIT e9d0177feafa3adad6ca90cadd44b50389ac4094
  OPTIONS
  "-DBUILD_SHARED_LIBS=${CALIBR8_BUILD_SHARED_LIBS}"
  "-DZLIB_COMPAT=ON"
  DEPENDENCIES
)
