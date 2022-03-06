# Specify the minimum required version of CMake to build the project.
cmake_minimum_required( VERSION 3.6 )
project( wiggly )
# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package( Houdini REQUIRED )
# Add a library with two source files.
set( library_name SOP_Wiggly )
add_library( ${library_name} SHARED
    src/SOP_Wiggly.C
    src/SOP_Wiggly.h
)
# Link against the Houdini libraries, and add required include directories and compile definitions.
target_link_libraries( ${library_name} Houdini )
# Configure several common target properties, such as its output directory.
houdini_configure_target( ${library_name} )