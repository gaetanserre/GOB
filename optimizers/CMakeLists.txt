cmake_minimum_required(VERSION 3.30)
project(optimizers)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fPIC -O3")

file(GLOB_RECURSE SRC_FILES src/*.cc)

include_directories(include ${NUMPY_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

# Include python headers
include_directories(${Python_INCLUDE_DIRS})

add_library(optimizers SHARED ${SRC_FILES})

link_libraries(optimizers ${Eigen_LIBRARIES} ${Python_LIBRARIES})
add_executable(example example.cc)