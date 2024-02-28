
#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Pass/AnalysisManager.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include <optional>

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"

#include <boost/graph/adjacency_list.hpp>

namespace toucan {

  struct PartitioningGraphNodeProperty {
    public:
    // LUTOpName opName;
    mlir::Operation *op;
    // weight is actually number of ops in this node
    uint32_t weight;
    bool isConstDecl;
  };

  // Note: use boost::vecS (std::vector) to ensure vertex_descriptor is integer and also incremental

  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, PartitioningGraphNodeProperty, boost::no_property, boost::no_property, boost::listS> PartitioningGraph;

  class IsLegalToucan4B {

  public:
    bool isToucanOnly;
    bool is4BOnly;
    bool isLegalToucan4B;

    IsLegalToucan4B(mlir::Operation *op);
  private:
    uint64_t chunkSize = 20000;
  };



  class DesignGraph {
  public:
    PartitioningGraph g;
    mlir::DenseSet<uint64_t> sinkVtxs;
    mlir::DenseSet<mlir::Operation*> constDeclVtxs;
    mlir::DenseMap<mlir::Operation*, uint64_t> opToId;
    
    DesignGraph(mlir::Operation *op, mlir::AnalysisManager &am);

    static bool opShouldRemoveInGraph(mlir::Operation *op);
  private:

  
  };


  class PartitionerBase {
    public:
    mlir::SmallVector<mlir::BitVector> partitions;
    mlir::SmallVector<uint32_t> vtxIdToPartId;
    // partition i -> level j -> elem k
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint64_t>>> partLevels;

    void levelizePartitions(DesignGraph &graph);
  };


  class NaivePartitioner: PartitionerBase {
    public:
    NaivePartitioner(mlir::Operation *op, mlir::AnalysisManager &am);
  };

  class RepCutPartitioner: PartitionerBase {
    public:
    RepCutPartitioner(mlir::Operation *op, mlir::AnalysisManager &am);
  };

}
