
#pragma once

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
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
#include <optional>

#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/CGToucanOpName.h"
#include "toucan/PartitioningGraph.h"


#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <filesystem>
#include <vector>

#define DESIGNGRAPH_EXGREAD_WEIGHT 1
#define DESIGNGRAPH_EXGWRITE_WEIGHT 1
#define DESIGNGRAPH_BREAK_IO_NOP_WEIGHT 0

#define GPU_THREAD_WARP_SIZE 32

namespace toucan {


  // MicroPart data structure
  class MicroPart {
    public:
    bool partIsValid;
    bool isNOPPart;
    uint32_t lineno, partId;

    mlir::DenseSet<uint32_t> nodes;
    // NodeId to ops ** in this part **
    mlir::DenseMap<uint32_t, uint32_t> nodeToOpCount;
    // LUT(and VecDecl), VecRead, RegRead, 
    CGToucanOPName opType;
    uint32_t totalOpCount;
    mlir::DenseSet<mlir::Value> inputValues, outputValueSet;
    mlir::SmallVector<mlir::Value> outputValues;


    // Only records levels! (unordered)
    // Valid for normal part
    mlir::SmallVector<mlir::SmallVector<uint32_t>> levels;
    mlir::DenseMap<uint32_t, uint32_t> nodeToLevel;
    mlir::DenseMap<uint32_t, mlir::SmallVector<mlir::Value>> nodeToInputVals;
    mlir::DenseMap<uint32_t, mlir::Value> nodeToOutputVal;
    mlir::DenseSet<uint32_t> dummyNodes;
    mlir::SmallVector<mlir::DenseSet<mlir::Value>> valuesUsedByEachLevel;

    // Valid for special ops
    mlir::SmallVector<mlir::Operation*> specialOps;

    // Valid for NOP part
    mlir::SmallVector<toucan::LUTOp> nops;



    // void schedule();
    void clear();
    void print() const;

    bool isRegularPart() const {return opType == toucan::CGToucanOPName::LUT;};

    void buildRegularLUTPart(const mlir::SmallVector<mlir::SmallVector<uint32_t>> &newNodesLevel);
    void buildSpecialPart(const CGToucanOPName vtxOpName, const mlir::SmallVector<mlir::Operation*> &rawOps);

    void mergeSpecialPartFromOtherParts(const mlir::SmallVector<std::shared_ptr<MicroPart>> &otherMPs);
    void buildNOPRegularLUTPart(mlir::SmallVector<toucan::LUTOp> &partNops);


    
    bool checkAndCollectIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId, const mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> outputVectorNopMap);


    private:
    void updateNodeToLevel();

    bool checkAndCollectRegularPartIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId, const mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> outputVectorNopMap);

    bool checkAndCollectSpecialPartIOValues(const PartitioningGraph &g, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId);
  };

}