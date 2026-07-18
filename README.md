# Toucan

Toucan is a **GPU-accelerated RTL simulator** that compiles hardware designs described in FIRRTL/Verilog into efficient GPU simulation kernels.

Toucan lowers CIRCT's HW/Comb/Seq/SV dialects through a custom MLIR dialect pipeline, partitions the design (RepCut + micro-partitioner), and generates CUDA simulation binary.

## Paper

If you use Toucan in your research, please cite:

> (*placeholder — paper citation will be added upon publication*)

## Prerequisites

| Dependency | Version |
|-----------|---------|
| CMake | >= 3.26 |
| Boost | (graph, filesystem) |
| MtKaHyPar | requires `tbb`, `hwloc` |
| toucan-micro-partitioner | (see `external/toucan-micro-partitioner`) |
| LLVM/MLIR | (built via CIRCT) |
| CIRCT | (see `external/circt`) |

## Building

Toucan uses Git submodules for its external dependencies:

```bash
git submodule update --init --recursive
```

Build all dependencies (LLVM/MLIR, CIRCT, partitioners):

```bash
cd external && ./build-deps.sh
```

Then build Toucan itself:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Pipeline Overview

```
FIRRTL/Verilog
    ↓ (firtool)
CIRCT CoreDialect (HW, Comb, Seq, SV)
    ↓ ToucanHigh  — Remove SV/OM, split mem ports, split registers, expand delays
    ↓ Toucan4B    — Lower to 4-bit wide signals (registers, memories, comb ops, vectors)
    ↓ ToucanFlattened — Flatten hierarchy, dedup registers, merge constants
    ↓ Code Generation — Partition, micro-partition, allocate values, serialize .bin design
GPU Simulation
```

## Usage

```bash
# Compile MLIR (CIRCT core dialect) to GPU simulation design
./toucan input.mlir -o ./output_dir -gpu

# Specify input level and dump intermediate MLIR
./toucan input.mlir -o ./output_dir -gpu --inputLevel=core --dump

# Tune partition size (ratio 0.01-10.0, default 1.0)
./toucan input.mlir -o ./output_dir -gpu --partSize 0.5

# Remove SV print/assert ops (no I/O during sim)
./toucan input.mlir -o ./output_dir -gpu --removeSVDialect
```

**Input levels:** `core` (default), `toucanHigh`, `toucan4B`, `toucanFlattened`

**Output files** (in `-o` directory):
- `GPUSimDesign.bin` — Compiled GPU simulation design
- `GPUSimSymbols.bin` — Symbol table
- `GPUSimIOSymbols.bin` — I/O symbol table
- `output.mlir` — (with `--dump`) Intermediate MLIR dump

The compiled design is run by the [Toucan GPU runtime](https://github.com/ucsc-vama/toucanGPURT).

## Directory Layout

- `include/toucan/` — Public headers and MLIR TableGen dialect definitions
- `lib/` — Library source (dialect, analysis, passes, simulation serialization)
- `tools/` — Executables (`toucan`, `firtool_test`)
- `external/` — Git submodules and build scripts for dependencies

## License

BSD 3-Clause. See `LICENSE`.
