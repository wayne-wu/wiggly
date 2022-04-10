cmake_minimum_required( VERSION 3.6 )

project( wiggly )

# CMAKE_PREFIX_PATH must contain the path to the toolkit/cmake subdirectory of
# the Houdini installation. See the "Compiling with CMake" section of the HDK
# documentation for more details, which describes several options for
# specifying this path.
list( APPEND CMAKE_PREFIX_PATH "C:/Program Files/Side Effects Software/Houdini 19.0.498/toolkit/cmake" )
list( APPEND GSL_ROOT_DIR "C:/dev/vcpkg/packages/gsl_x64-windows" )

# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package( Houdini REQUIRED )
find_package( GSL REQUIRED)

set( library_name SOP_Wiggly )

# Code generation for the embedded DS file in SOP_Star.C.
houdini_generate_proto_headers( FILES SOP_Wiggly.C )

# Add a library and its source files.
add_library( ${library_name} SHARED
    SOP_Wiggly.C
    SOP_Wiggly.h
    Wiggly.h
    Wiggly.C
)

add_subdirectory(C:/dev/dlib-19.23 dlib_build)

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries( ${library_name} Houdini )
target_link_libraries( ${library_name} GSL::gsl GSL::gslcblas)
target_link_libraries( ${library_name} dlib::dlib )

# Include ${CMAKE_CURRENT_BINARY_DIR} for the generated header.
target_include_directories( ${library_name} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories( ${library_name} PRIVATE Eigen)
target_include_directories( ${library_name} PRIVATE unsupported)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target( ${library_name} )
