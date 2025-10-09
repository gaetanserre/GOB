cmake_minimum_required(VERSION 3.28)
project(optimizers)

# Detect platform
if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
elseif(UNIX)
  set(DARWIN TRUE)
else()
  set(WINDOWS TRUE)
endif()

file(GLOB_RECURSE SRC_FILES src/*.cc src/*.cpp src/*.c)
file(GLOB_RECURSE GLPK_SRC_FILES glpk-5.0/src/*.c)
file(GLOB_RECURSE GLPK_HEADERS glpk-5.0/src/*.h)

set(GLPK_INCLUDE_DIRS "")
foreach(hfile IN LISTS GLPK_HEADERS)
    get_filename_component(dir ${hfile} DIRECTORY)
    list(APPEND GLPK_INCLUDE_DIRS ${dir})
endforeach()
list(REMOVE_DUPLICATES GLPK_INCLUDE_DIRS)

find_package(Eigen3 3.4.0 REQUIRED)
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

set(Python_EXECUTABLE "python" CACHE STRING "Path to the Python executable")

execute_process(
  COMMAND
  ${Python_EXECUTABLE} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
  OUTPUT_VARIABLE
  Python_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS} CACHE STRING "" FORCE)
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")

include_directories(include ${GLPK_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

if(WINDOWS)
  add_library(glpk_obj SHARED ${GLPK_SRC_FILES})
  set_target_properties(glpk_obj PROPERTIES LINKER_LANGUAGE C)
  target_include_directories(glpk_obj PUBLIC ${GLPK_INCLUDE_DIRS})
  target_compile_options(glpk_obj PRIVATE -O3 -fPIC -std=c99)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")

  # Find Python library
  find_package(Python COMPONENTS Development.Module REQUIRED)
  message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
  add_library(${EXT_NAME} SHARED ${SRC_FILES} ${CYTHON_CPP_FILE} $<TARGET_OBJECTS:glpk_obj>)
  target_link_libraries(${EXT_NAME} ${Python_LIBRARIES})
else()
  add_library(glpk_obj SHARED ${GLPK_SRC_FILES})
  set_target_properties(glpk_obj PROPERTIES LINKER_LANGUAGE C)
  target_include_directories(glpk_obj PUBLIC ${GLPK_INCLUDE_DIRS})
  target_compile_options(glpk_obj PRIVATE -O3 -fPIC -std=c99)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "-fPIC -O3")
  add_library(${EXT_NAME} SHARED ${SRC_FILES} ${CYTHON_CPP_FILE} $<TARGET_OBJECTS:glpk_obj>)
endif()

# Set macOS-specific linker flags
if(APPLE)
  set_target_properties(
    ${EXT_NAME}
    PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup"
  )
endif()