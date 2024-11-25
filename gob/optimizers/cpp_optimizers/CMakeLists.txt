cmake_minimum_required(VERSION 3.30)
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

if(DARWIN)
  set(LIB_EXT "dylib")
elseif(LINUX)
  set(LIB_EXT "so")
else()
  set(LIB_EXT "dll")
endif()

file(GLOB_RECURSE SRC_FILES src/*.cc src/*.cpp src/*.c)

find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

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