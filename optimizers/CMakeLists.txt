cmake_minimum_required(VERSION 3.28)
project(optimizers)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fPIC -O3")

file(GLOB_RECURSE SRC_FILES src/*.cc)

include_directories(include)

find_package(Eigen3 REQUIRED)

add_library(optimizers SHARED ${SRC_FILES})

link_libraries(optimizers ${Eigen_LIBRARIES})
add_executable(example src/example.cc)