#!/bin/bash

set -e

if [[ "$BUILD_TYPE" == "" ]]; then
  BUILD_TYPE="DEBUG"
fi

INSTALL_PREFIX=$(realpath ../install)
[ -d $INSTALL_PREFIX ] || mkdir $INSTALL_PREFIX



# build micro partitioner
pushd toucan-micro-partitioner

[ -d build ] || mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=RELEASE
make -j$(nproc) install

popd

# Create launcher for toucan-micro-partitioner (python version)
EXTERNAL_LIB_DIR=$(realpath ./)
BIN_DIR="$INSTALL_PREFIX/bin"
LAUNCHER_SCRIPT="$BIN_DIR/toucan-mpart"
PYTHON_SCRIPT="$EXTERNAL_LIB_DIR/toucan-micro-partitioner/MicroPartitioner.py"

# Ensure directories exist
mkdir -p "$BIN_DIR"

# Verify Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!" >&2
    exit 1
fi

# Create launcher
cat > "$LAUNCHER_SCRIPT" << EOF
#!/bin/sh
# Launcher for toucan-micro-partitioner

exec python3 "$PYTHON_SCRIPT" "\$@"
EOF

# Set permissions
chmod +x "$LAUNCHER_SCRIPT"




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
