#!/bin/bash

if [[ "$BUILD_TYPE" == "" ]]; then
  BUILD_TYPE="DEBUG"
fi

INSTALL_PREFIX=$(realpath ../install)
[ -d $INSTALL_PREFIX ] || mkdir $INSTALL_PREFIX


# build KaHyPar
pushd kahypar

git apply ../KaHyPar_cmake.diff || true
[ -d build ] || mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j$(nproc) install

popd

# build rcp
pushd RepCut-Partitioner

[ -d build ] || mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j$(nproc) install

popd

cd circt

# build llvm
pushd llvm

[ -d build ] || mkdir build
cd build
cmake ../llvm \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j$(nproc) install

popd

# build circt
[ -d build ] || mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DVERILATOR_DISABLE=ON \
    -DCIRCT_LLHD_SIM_ENABLED=OFF
make -j$(nproc) install