cmake_minimum_required(VERSION 3.13.4)

project(toucan 
    VERSION 0.0.0 
    LANGUAGES CXX)

# Generate compile_commands.json for vscode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


# set(CIRCT_CMAKE_DIR "")
# set(MLIR_CMAKE_DIR "")
# set(LLVM_CMAKE_DIR "")

# list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)


set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)



include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(AddCIRCT)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CIRCT_MAIN_INCLUDE_DIR} ${CIRCT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(external)
# add_subdirectory(include)
# add_subdirectory(lib)
add_subdirectory(tools)

add_executable(toucan main.cpp)