cmake_minimum_required(VERSION 3.10.0)
project(nodesig)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

include_directories(lib)

set(SOURCE_FILES src/Graph.cpp src/Model.cpp)
set(LIB_FILES lib/Graph.h lib/Model.h lib/Utilities.h)

add_executable(${CMAKE_PROJECT_NAME} main.cpp ${SOURCE_FILES} ${LIB_FILES})

################ OpenMP ################
find_package(OpenMP REQUIRED)


target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

