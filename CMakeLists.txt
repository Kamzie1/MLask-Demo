cmake_minimum_required(VERSION 3.28)
project(mlask-demo)

include(FetchContent)
message("importing MLask...")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(mlask
    GIT_REPOSITORY https://github.com/Kamzie1/MLask.git
    GIT_TAG 1.0.4
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(mlask)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(demo main.cpp)
target_compile_features(demo PRIVATE cxx_std_23)
target_link_libraries(demo PUBLIC mlask)

