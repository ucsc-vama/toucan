
#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Value.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"


#include <cstdint>
#include <optional>

#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/PartitioningGraph.h"

#include "toucan/MicroPartitioner.h"

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <filesystem>
#include <vector>

namespace toucan {
  class MicroPartLocalValueAllocator {
    public:
    size_t numTotalValSize;
    // numConsts = compactConstValPool.size()
    size_t numConsts;
    size_t numOutputVals;
    size_t numInputVals;

    // Value pool:
    // const values, input values, output values, tempories 
    mlir::DenseMap<mlir::Value, uint32_t> valToValId;
    mlir::SmallVector<mlir::Value> valuePool;
    // Use this as the real const pool
    // still, val 0 is always 0
    mlir::SmallVector<uint8_t> compactConstValPool;

    void allocateLocalValues();
    

    // uint32_t getValId(mlir::Value);

    void collectValueLifetime(const PartitioningGraph &graph, const MicroPartitioner &mpartitioner);

    // Populate const vals and RegWrite/ExchangeWrite. Those values are pinned
    void populateInitialPinnedVals(const PartitioningGraph &graph, const mlir::DenseMap<mlir::Value, uint32_t> constValToRawValue, const MicroPartitioner &mpartitioner);


    private:
    struct ValueLifeTime {
      uint32_t start;
      uint32_t end;
    };


    mlir::DenseSet<mlir::Value> pinnedInputVals, pinnedOutputVals, constVals;

    mlir::DenseMap<mlir::Value, ValueLifeTime> valToLifeTime;
    mlir::DenseMap<mlir::Value, uint32_t> vecValToLength;

    size_t totalLevels;

  };
}