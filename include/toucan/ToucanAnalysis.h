
#pragma once

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include "circt/Dialect/HW/HWDialect.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "toucan/ToucanCodeGenInfo.h"
#include "toucan/ToucanConfigs.h"

#include "toucan/MicroPartitioner.h"

#include <boost/graph/adjacency_list.hpp>

#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>

#define DESIGNGRAPH_EXGREAD_WEIGHT 1
#define DESIGNGRAPH_EXGWRITE_WEIGHT 1
#define DESIGNGRAPH_BREAK_IO_NOP_WEIGHT 0

#define GPU_THREAD_WARP_SIZE 32

namespace toucan {




  class IsLegalToucan4B {

  public:
    bool isToucanOnly;
    bool is4BOnly;
    bool isLegalToucan4B;

    IsLegalToucan4B(mlir::Operation *op);
  private:
    uint32_t chunkSize = 20000;
  };

  size_t getExtraAlignmentSpace(size_t valSize, size_t alignment);


  class DesignGraph {
  public:
    PartitioningGraph g;
    mlir::DenseMap<mlir::Operation*, uint32_t> opToId;
    mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regs;
    
    DesignGraph(mlir::Operation *op, mlir::AnalysisManager &am);

    static bool opShouldRemoveInGraph(mlir::Operation *op);
  private:
  
  };



  class LocalValueAllocator {
    public:
    size_t numTotalValSize;
    // numConsts = compactConstValPool.size()
    size_t numConsts;
    size_t numOutputVals;
    size_t numInputVals;

    mlir::DenseMap<mlir::Value, uint32_t> valToValId;
    // Use this as the real const pool
    // still, val 0 is always 0
    mlir::SmallVector<uint8_t> compactConstValPool;

    void allocateLocalValues();
    

    // uint32_t getValId(mlir::Value);

    void collectValueLifetime(PartitioningGraph &regionGraph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partLevels, const mlir::SmallVector<CGExchangeValueMetaInfo> &exchangePool);

    // Populate const vals and RegWrite/ExchangeWrite. Those values are pinned
    void populateInitialPinnedVals(PartitioningGraph &regionGraph, const mlir::DenseMap<mlir::Value, uint32_t> constValToRawValue, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partLevels, const mlir::SmallVector<CGExchangeValueMetaInfo> &exchangePool);


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
