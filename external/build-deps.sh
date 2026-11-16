#!/bin/bash

set -e

if [[ "$BUILD_TYPE" == "" ]]; then
  BUILD_TYPE="RELEASE"
fi

# Resolve the install prefix relative to this script's location and create it if
# missing. install/ is gitignored, so it does not exist on a fresh checkout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="$SCRIPT_DIR/../install"
mkdir -p "$INSTALL_PREFIX"
INSTALL_PREFIX="$(cd "$INSTALL_PREFIX" && pwd)"



# build micro partitioner
pushd toucan-micro-partitioner

[ -d build ] || mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=RELEASE
make -j$(nproc) install

popd

# build rcp
pushd RepCut-Partitioner

[ -d build ] || mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=RELEASE
make -j$(nproc) install

popd

# build MtKaHyPar
pushd mt-kahypar

[ -d build ] || mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=RELEASE
make -j$(nproc) MtKaHyPar

cp ./mt-kahypar/application/MtKaHyPar $INSTALL_PREFIX/bin/

popd



cd circt

# Temporary fix:
# The pinned LLVM/CIRCT sources use uint*_t in some headers (e.g.
# llvm/ADT/SmallVector.h, lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h) without
# including <cstdint>. Older toolchains provided it transitively, but GCC 16+ no
# longer does, so force-include it for the LLVM and CIRCT builds.
CXX_FLAGS_CSTDINT="-include cstdint"

# build llvm
pushd llvm

[ -d build ] || mkdir build
cd build
cmake ../llvm \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS_CSTDINT" \
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
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS_CSTDINT" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DVERILATOR_DISABLE=ON \
    -DCIRCT_LEC_DISABLE=ON \
    -DCIRCT_LLHD_SIM_ENABLED=OFF
make -j$(nproc) install
