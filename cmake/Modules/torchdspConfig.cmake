if(NOT PKG_CONFIG_FOUND)
    INCLUDE(FindPkgConfig)
endif()
PKG_CHECK_MODULES(PC_TORCHDSP torchdsp)

FIND_PATH(
    TORCHDSP_INCLUDE_DIRS
    NAMES torchdsp/api.h
    HINTS $ENV{TORCHDSP_DIR}/include
        ${PC_TORCHDSP_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    TORCHDSP_LIBRARIES
    NAMES gnuradio-torchdsp
    HINTS $ENV{TORCHDSP_DIR}/lib
        ${PC_TORCHDSP_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/torchdspTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TORCHDSP DEFAULT_MSG TORCHDSP_LIBRARIES TORCHDSP_INCLUDE_DIRS)
MARK_AS_ADVANCED(TORCHDSP_LIBRARIES TORCHDSP_INCLUDE_DIRS)
