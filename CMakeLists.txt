
project(dasilva_invariants)
find_package(F2PY REQUIRED)

# Boilerplate
message("conda prefix is: $ENV{CONDA_PREFIX}")
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)

file(GLOB pythonh $ENV{CONDA_PREFIX}/include/python*/Python.h)
message("Python.h is ${pythonh}")
get_filename_component(pythonh_dirname ${pythonh} DIRECTORY)
message("Python.h dirname is ${pythonh_dirname}")

file(GLOB arrayobjecth $ENV{CONDA_PREFIX}/lib/python*/site-packages/numpy/core/include/numpy/arrayobject.h)
list(GET arrayobjecth -1 arrayobjecth)
message("arrayobject.h is ${arrayobjecth}")
get_filename_component(arrayobjecth_dirname ${arrayobjecth} DIRECTORY)
message("arrayobject.h dirname is ${arrayobjecth_dirname}")

include_directories(${pythonh_dirname} ${arrayobjecth_dirname})

# Function to copmile $tagname.f and $tagname_interface.f into python module
# a la "from dasilva_invariants._fortran import $tagname".
function(add_fortran_item tagname functions)
  set(custom_target_name "install_${tagname}") 
  set(f2py_module_name ${tagname})
  set(fortran_src_file
    ${CMAKE_CURRENT_SOURCE_DIR}/dasilva_invariants/_fortran/${tagname}.f
    ${CMAKE_CURRENT_SOURCE_DIR}/dasilva_invariants/_fortran/${tagname}_interface.f
  )
  set(generated_module_file
    ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}.so
  )
  add_custom_target(${custom_target_name} ALL DEPENDS ${generated_module_file})  

  # This is bad practice, but life is too short to figure out CMake
  add_custom_command(
    OUTPUT ${generated_module_file}
    BUILD
    COMMAND ${F2PY_EXECUTABLE}
    -m ${f2py_module_name}
    -c
    ${fortran_src_file}
    only: ${functions} :
    && mv ${f2py_module_name}*.so ${generated_module_file}
    && mkdir -p ${CMAKE_INSTALL_PREFIX}/dasilva_invariants/_fortran
    && cp ${generated_module_file} ${CMAKE_INSTALL_PREFIX}/dasilva_invariants/_fortran
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endfunction()

add_fortran_item("geopack2008" "recalc force_dipole_tilt dipnumpy")
add_fortran_item("t96" "t96numpy")
add_fortran_item("ts05" "ts05numpy")

add_custom_target(install ALL DEPENDS install_geopack2008 install_t96 install_ts05)
