cmake_minimum_required(VERSION 3.30)
project(optimizers)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-fPIC -O3")

file(GLOB_RECURSE SRC_FILES src/*.cc)

find_package(Eigen3 REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

# Include python headers
include_directories(include ${NUMPY_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

add_library(optimizers SHARED ${SRC_FILES})

link_libraries(optimizers ${Eigen_LIBRARIES} ${Python_LIBRARIES} ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcmaes.so)
add_executable(example example.cc)