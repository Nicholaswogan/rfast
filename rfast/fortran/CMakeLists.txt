# Fortran that f2py needs to parse
set(F2PY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lblabc_input.f95)

set(f2py_module_name "_lblabc_input") # name of the python module

# Most of the stuff below is boilerplate CMake required
# to make a f2py library.

set(generated_module_file ${f2py_module_name}${PYTHON_EXTENSION_MODULE_SUFFIX})

add_custom_target(${f2py_module_name} ALL
DEPENDS ${F2PY_SOURCES}
)

add_custom_command(
OUTPUT "${f2py_module_name}module.c"
COMMAND ${F2PY_EXECUTABLE}
  -m ${f2py_module_name}
  ${F2PY_SOURCES}
  --lower
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
DEPENDS ${F2PY_SOURCES}
)

add_library(
  ${generated_module_file} MODULE
  "${f2py_module_name}module.c"
  "${F2PY_INCLUDE_DIR}/fortranobject.c"
   ${F2PY_SOURCES}
)

target_include_directories(
  ${generated_module_file} PUBLIC
  "${CMAKE_BINARY_DIR}/modules"
  ${F2PY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

set_target_properties(${generated_module_file} PROPERTIES SUFFIX "")
set_target_properties(${generated_module_file} PROPERTIES PREFIX "")    

if (UNIX)
  if (APPLE)
    set_target_properties(${generated_module_file} PROPERTIES
    LINK_FLAGS  '-Wl,-dylib,-undefined,dynamic_lookup')
  else()
    set_target_properties(${generated_module_file} PROPERTIES
  LINK_FLAGS  '-Wl,--allow-shlib-undefined')
  endif()
endif()

# Install location depends on ${SKBUILD} option.
if (SKBUILD)
  install(TARGETS ${generated_module_file} DESTINATION rfast)
else()
  install(TARGETS ${generated_module_file} DESTINATION ${CMAKE_SOURCE_DIR}/rfast)
endif()