cmake_minimum_required(VERSION 3.30)
project(optimizers)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fPIC -O3")

file(GLOB_RECURSE SRC_FILES src/*.cc)

include_directories(include)

find_package(Eigen3 REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)
find_package(Boost REQUIRED COMPONENTS python)

# Include python headers
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

message(PROJECT_SOURCE_DIR="${Python_RUNTIME_LIBRARY_DIRS}")

add_library(optimizers SHARED ${SRC_FILES})

link_libraries(optimizers ${Eigen_LIBRARIES} ${Boost_LIBRARIES} ${Python_LIBRARIES})
add_executable(example example.cc)