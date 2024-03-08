
#pragma once

#include "circt/Dialect/HW/HWOps.h"
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
#include "llvm/Support/ErrorHandling.h"

#include "circt/Dialect/HW/HWDialect.h"

#include <cstdint>
#include <optional>

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>

namespace toucan {

  enum class CGToucanOPName {
    ConstDecl,
    LUT,
    VecRead,
    VecDecl,
    Print,
    Stop,
    RegRead,
    RegWrite,
    MemRead,
    MemWrite,
    ShouldNotAppear
    // Constant,
    // ConstVec
  };

  struct PartitioningGraphNodeProperty {
    public:
    // LUTOpName opName;
    mlir::Operation *op;
    // weight is actually number of ops in this node
    uint32_t weight;
    
    CGToucanOPName toucanOpName;
  };

  // Note: use boost::vecS (std::vector) to ensure vertex_descriptor is integer and also incremental

  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, PartitioningGraphNodeProperty, boost::no_property, boost::no_property, boost::listS> PartitioningGraph;



  struct CGValueMetaInfo {
    bool isConst;
    bool isPlaceholder;
    // value is valid if isConst
    uint8_t value;
    mlir::Operation *definingOp;

    uint32_t levelId;
    uint32_t opId;

    std::optional<mlir::StringRef> namehint;
    uint32_t fragment_id;
  };


  struct CGLUTOpMetaInfo {
    uint8_t lutId;
    // assert size == 3
    uint32_t op0;
    uint32_t op1;
    uint32_t op2;

    uint8_t numOprands;

    uint32_t result;
  };
  struct CGVectorReadOpMetaInfo {
    uint32_t vecBase;
    uint32_t vecLength;
    
    // max addr width: 16
    uint32_t index0;
    uint32_t index1;
    uint32_t index2;
    uint32_t index3;
    uint32_t outRangeValue;
    // Note: This offset is a static value!!!!
    uint16_t offset;

    uint32_t result;
  };

  struct CGPrintOpMetaInfo {
    uint32_t en;
    // Index of target string in string pool
    uint32_t msg;
  };
  struct CGStopOpMetaInfo {
    uint32_t en;
  };
  struct CGRegReadOpMetaInfo {
    // reg: index in global regPool
    uint32_t reg;
    uint32_t result;
  };
  struct CGRegWriteOpMetaInfo {
    uint32_t reg;
    uint32_t dat;
  };
  struct CGMemReadOpMetaInfo {
    bool hasMultipleWriter;
    // Should do boundary check
    uint32_t memDepth;
    
    uint64_t memBase;

    // max addr width: 32
    uint32_t addr0;
    uint32_t addr1;
    uint32_t addr2;
    uint32_t addr3;
    uint32_t addr4;
    uint32_t addr5;
    uint32_t addr6;
    uint32_t addr7;

    uint32_t result;
  };
  struct CGMemWriteOpMetaInfo {
    bool hasMultipleWriter;
    // should do boundary check
    uint32_t memDepth;

    uint64_t memBase;

    // max addr width: 32
    uint32_t addr0;
    uint32_t addr1;
    uint32_t addr2;
    uint32_t addr3;
    uint32_t addr4;
    uint32_t addr5;
    uint32_t addr6;
    uint32_t addr7;

    uint32_t dat;
    uint32_t en;
  };


  struct CGOpMetaInfo {
    CGToucanOPName opName;
    
    union {
      // top level
      CGMemReadOpMetaInfo memRead;
      CGRegReadOpMetaInfo regRead;

      // middle levels, exec
      CGLUTOpMetaInfo lut;
      CGVectorReadOpMetaInfo vec;

      // last level
      CGMemWriteOpMetaInfo memWrite;
      CGRegWriteOpMetaInfo regWrite;
      CGPrintOpMetaInfo print;
      CGStopOpMetaInfo stop;
    };

    std::optional<mlir::StringRef> namehint;
    uint32_t fragment_id;

    mlir::Operation *op;
    uint32_t vtxId;

    void setResult(uint32_t result) {
      switch (opName) {
        case CGToucanOPName::LUT: {
          lut.result = result;
          return;
        }
        case CGToucanOPName::VecRead: {
          vec.result = result;
          return;
        }
        case CGToucanOPName::RegRead: {
          regRead.result = result;
          return;
        }
        case CGToucanOPName::MemRead: {
          memRead.result = result;
          return;
        }

        // ConstDecl & VecDecl: Should not have any op with such type
        case CGToucanOPName::ConstDecl:
        case CGToucanOPName::VecDecl:

        case CGToucanOPName::ShouldNotAppear:
        case CGToucanOPName::Print:
        case CGToucanOPName::Stop:
        case CGToucanOPName::RegWrite:
        case CGToucanOPName::MemWrite: {
          llvm::dbgs() << "Error: should not have result\n";
          assert(false);
        }
      }
      llvm_unreachable("Should not reach here");
    }
    uint32_t getResult() {
      switch (opName) {
        case CGToucanOPName::LUT: return lut.result;
        case CGToucanOPName::VecRead: return vec.result;
        case CGToucanOPName::RegRead: return regRead.result;
        case CGToucanOPName::MemRead: return memRead.result;

        // ConstDecl & VecDecl: Should not have any op with such type
        case CGToucanOPName::ConstDecl:
        case CGToucanOPName::VecDecl:

        case CGToucanOPName::ShouldNotAppear:
        case CGToucanOPName::Print:
        case CGToucanOPName::Stop:
        case CGToucanOPName::RegWrite:
        case CGToucanOPName::MemWrite: {
          llvm::dbgs() << "Error: should not have result\n";
          assert(false);
        }
      }
      llvm_unreachable("Should not reach here");
    }
    bool hasResult() {
      switch (opName) {
        case CGToucanOPName::LUT:
        case CGToucanOPName::VecRead:
        case CGToucanOPName::RegRead:
        case CGToucanOPName::MemRead: return true;

        default: return false;
      }
    }
  };

  struct CGRegMetaInfo {
    std::optional<mlir::StringRef> namehint;
    uint32_t bitWidth;
    uint32_t fragment_id;
    bool isPadding;
  };

  struct CGMemMetaInfo {
    std::optional<mlir::StringRef> namehint;
    uint32_t bitWidth;
    uint64_t memDepth;
    uint32_t fragment_id;
    bool hasMultipleWriter;
    uint64_t memBase;
  };

  struct CGLayerValueStatistics {
    // only in first level
    uint32_t numRegReads;
    // only in middle levels
    uint32_t numMemReads;
    uint32_t numVecReads;
    uint32_t numLuts;
    // only in last level
    uint32_t numRegWrites;
    uint32_t numMemWrites;
    uint32_t numPrints;
    uint32_t numStops;
  };

  struct CGOpStatistics {
    uint32_t numRegReads;
    uint32_t numMemReads;
    uint32_t numVecReads;
    uint32_t numLuts;
    uint32_t numLutNops;
    uint32_t numRegWrites;
    uint32_t numMemWrites;
    uint32_t numPrints;
    uint32_t numStops;
  };

  // Information needed for code gen, each level
  struct CGPartitionMetaInfo {
    // Values generated by each layer. Will also be used as input of next level
    mlir::SmallVector<CGValueMetaInfo> valuePool;
    // Beginning N values are constant
    uint32_t numConstsInValuePool;
    // Operator pool in each level
    mlir::SmallVector<mlir::SmallVector<CGOpMetaInfo>> opPool;
    mlir::DenseMap<mlir::Value, uint32_t> valueToValId;

    mlir::SmallVector<CGLayerValueStatistics> opStatisticsPerLevel;
    CGOpStatistics opStatistics;
  };

  struct CGInfo {
    // Global register region. 
    mlir::SmallVector<CGRegMetaInfo> regPool;
    // Memories
    mlir::SmallVector<CGMemMetaInfo> memPool;
    uint64_t totalMemSize;

    mlir::DenseMap<mlir::StringRef, uint32_t> printStrings;

    // debug info
    // name -> (fragment 0, 1, 2, ...)
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<uint32_t>> regDebugInfo;
    // name -> ((part, valId), (part, valId), ..)
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<std::tuple<uint32_t, uint32_t>>> signalDebugInfo;
    // name -> (start pos, bit width, length)
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<uint32_t>> memDebugInfo;

    // For developing purpose
    mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> toucanRegToId;
    mlir::DenseMap<mlir::TypedValue<toucan::MemType>, uint32_t> toucanMemToId;

    mlir::SmallVector<CGPartitionMetaInfo, 4> partitionInfo;
  };

  class IsLegalToucan4B {

  public:
    bool isToucanOnly;
    bool is4BOnly;
    bool isLegalToucan4B;

    IsLegalToucan4B(mlir::Operation *op);
  private:
    uint32_t chunkSize = 20000;
  };



  class DesignGraph {
  public:
    PartitioningGraph g;
    // mlir::DenseSet<uint32_t> sinkVtxs;
    // mlir::DenseSet<mlir::Operation*> constDeclVtxs;
    mlir::DenseMap<mlir::Operation*, uint32_t> opToId;

    // mlir::SmallVector<toucan::DefConstVectorOp*> constVecOps;
    
    DesignGraph(mlir::Operation *op, mlir::AnalysisManager &am);

    static bool opShouldRemoveInGraph(mlir::Operation *op);
  private:

  
  };


  class PartitionerNCodeGenBase {
  public:
    mlir::SmallVector<mlir::BitVector> partitions;
    mlir::SmallVector<uint32_t> vtxIdToPartId;
    // partition i -> level j -> elem k
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> partLevels;

    CGInfo codeGenInfo;

    void levelizePartitions(DesignGraph &graph);

    void generateMemoryLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);

    void fillDebugInfo();

  private:

    void collectConstant(DesignGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId);
    // void collectPrintMsgs(DesignGraph &graph);
    void generateRegMemLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);
    void collectPrintString(DesignGraph &graph);
  };


  class NaivePartitioner: PartitionerNCodeGenBase {
    public:
    NaivePartitioner(mlir::Operation *op, mlir::AnalysisManager &am);
  };

  class RepCutPartitioner: PartitionerNCodeGenBase {
    public:
    RepCutPartitioner(mlir::Operation *op, mlir::AnalysisManager &am);
  };

}
