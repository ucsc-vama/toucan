#!/bin/bash

if [[ "$INSTALL_PREFIX" == "" ]]; then
  INSTALL_PREFIX=$(realpath ../install)
  [ -d $INSTALL_PREFIX ] || mkdir $INSTALL_PREFIX
fi

cd circt

# build llvm
pushd llvm

[ -d build ] || mkdir build
cd build
cmake -G Ninja ../llvm \
    -D-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
ninja
# ninja check-mlir

popd

[ -d build ] || mkdir build
cd build
cmake -G Ninja .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DESI_COSIM=OFF -DESI_CAPN=OFF \
    -DVERILATOR_DISABLE=ON
ninja
# ninja check-circt