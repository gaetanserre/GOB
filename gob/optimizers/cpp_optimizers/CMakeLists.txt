cmake_minimum_required(VERSION 3.28)
project(optimizers)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O3 -fPIC")

# Detect platform
if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
elseif(UNIX)
  set(DARWIN TRUE)
else()
  set(WINDOWS TRUE)
endif()

file(GLOB_RECURSE SRC_FILES src/*.cc src/*.cpp src/*.c)

find_package(Eigen3 REQUIRED)

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

include_directories(include include/glpk ${EIGEN3_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

add_library(${EXT_NAME} SHARED ${SRC_FILES} ${CYTHON_CPP_FILE})

# Set macOS-specific linker flags
if(APPLE)
  set_target_properties(
    ${EXT_NAME}
    PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup"
  )
endif()