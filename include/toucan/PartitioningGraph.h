
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "mlir/IR/Value.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"


#include <cstdint>
#include <optional>

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <filesystem>

namespace toucan {
  enum class CGToucanOPName {
    ConstDecl,
    LUT,
    VecRead,
    VecDecl,
    VecLogic,
    VecArith,
    VecStaticRead,
    Print,
    Stop,
    RegRead,
    RegWrite,
    MemRead,
    MemWrite,
    ShouldNotAppear,
    // Note: Those nodes should be removed from graph. They do not emit code
    Dummy_DefReg,
    Dummy_DefMem,
    // Note: Exchange between regions in MultiRegionScheduler.
    ExchangeRead,
    ExchangeWrite
    // Constant,
    // ConstVec
  };

  constexpr int getMaxEnumValForCGToucanOPName() {return static_cast<int>(CGToucanOPName::ExchangeWrite);};

  std::string stringifyCGToucanOPName(CGToucanOPName val);

  struct PartitioningGraphNodeProperty {
    public:
    // LUTOpName opName;
    mlir::Operation *op;
    // weight is actually number of ops in this node
    uint32_t weight;
    uint32_t opCount;
    uint32_t exchangeValId;
    
    CGToucanOPName toucanOpName;
  };

  // Note: use boost::vecS (std::vector) to ensure vertex_descriptor is integer and also incremental

  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, PartitioningGraphNodeProperty, boost::no_property, boost::no_property, boost::listS> PartitioningGraph;

  // typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS, PartitioningGraphNodeProperty, boost::no_property> PartitioningGraph;

  void mergeVerticies(uint32_t dst, const mlir::SmallVector<uint32_t> &toMerge, PartitioningGraph &g);

  bool partitioningGraphHasCycle(const PartitioningGraph &graph);

}