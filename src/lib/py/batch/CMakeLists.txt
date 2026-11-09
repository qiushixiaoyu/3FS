cmake_minimum_required(VERSION 3.20)

# Also buildable against an installed API library without rebuilding 3FS.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hf3fs_py_batch LANGUAGES CXX)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  find_path(HF3FS_API_INCLUDE_DIR hf3fs_usrbio.h
    HINTS "${HF3FS_API_ROOT}/include" REQUIRED)
  find_library(HF3FS_API_LIBRARY hf3fs_api_shared
    HINTS "${HF3FS_API_ROOT}/lib" REQUIRED)
  add_library(hf3fs_api_shared SHARED IMPORTED)
  set_target_properties(hf3fs_api_shared PROPERTIES
    IMPORTED_LOCATION "${HF3FS_API_LIBRARY}")
else()
  set(HF3FS_API_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/lib/api")
endif()

pybind11_add_module(hf3fs_py_batch batch_io.cc)
target_compile_features(hf3fs_py_batch PRIVATE cxx_std_17)
target_include_directories(hf3fs_py_batch PRIVATE "${HF3FS_API_INCLUDE_DIR}")
target_link_libraries(hf3fs_py_batch PRIVATE hf3fs_api_shared)
install(TARGETS hf3fs_py_batch LIBRARY DESTINATION lib)
