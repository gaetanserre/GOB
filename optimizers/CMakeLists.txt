cmake_minimum_required(VERSION 3.30)
project(optimizers)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-fPIC -O3")

file(GLOB_RECURSE SRC_FILES src/*.cc)

find_package(Eigen3 REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

include_directories(include ${EIGEN3_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

link_libraries(${Eigen_LIBRARIES} ${Python_LIBRARIES} ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcmaes.so)

add_library(${EXT_NAME} SHARED ${SRC_FILES} ${CYTHON_CPP_FILE})

add_executable(example example.cc)