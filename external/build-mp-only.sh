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
