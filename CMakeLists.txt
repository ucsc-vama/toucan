cmake_minimum_required(VERSION 3.13.4)

cmake_policy(SET CMP0116 NEW)
# Introduced after cmake 3.30. Both NEW and OLD works, prefer NEW behavior as OLD behavior is deprecated.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(toucan 
    VERSION 0.0.0 
    LANGUAGES C CXX)

# Generate compile_commands.json for vscode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Remove this warning introduced by mlir
  add_compile_options(-Wno-ambiguous-reversed-operator)
endif()

list(APPEND CMAKE_PREFIX_PATH "./install/")

find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

find_package( Boost REQUIRED COMPONENTS graph )
include_directories( ${Boost_INCLUDE_DIRS} )


set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)



get_filename_component(MLIR_DIR_ABSOLUTE ${MLIR_DIR} REALPATH)
list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR_ABSOLUTE})

get_filename_component(CIRCT_DIR_ABSOLUTE ${CIRCT_DIR} REALPATH)
list(APPEND CMAKE_MODULE_PATH ${CIRCT_DIR_ABSOLUTE})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(AddCIRCT)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CIRCT_MAIN_INCLUDE_DIR} ${CIRCT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(external)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# add_executable(toucan main.cpp)