
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
    Print,
    Stop,
    RegRead,
    RegWrite,
    MemRead,
    MemWrite,
    ShouldNotAppear,
    // Note: Exchange between 2 regions in TwoRegionScheduler.
    ExchangeRead,
    ExchangeWrite
    // Constant,
    // ConstVec
  };

  std::string stringifyCGToucanOPName(CGToucanOPName val);

  struct PartitioningGraphNodeProperty {
    public:
    // LUTOpName opName;
    mlir::Operation *op;
    // weight is actually number of ops in this node
    uint32_t weight;
    uint32_t exchangeValId;
    
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
    uint8_t bitWidth;

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
    // Note: vecLength is static
    uint16_t vecLength;
    // Note: This offset is a static value!!!!
    uint16_t offset;
    
    // max addr width: 16
    uint32_t index0;
    uint32_t index1;
    uint32_t index2;
    uint32_t index3;
    uint32_t outRangeValue;

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
    // static
    bool hasMultipleWriter;
    // static
    uint32_t memDepth;
    // static
    uint64_t memBase;

    uint32_t en;
    uint32_t addrVec;

    uint32_t result;
  };
  struct CGMemWriteOpMetaInfo {
    bool hasMultipleWriter;
    uint32_t memDepth;
    uint64_t memBase;

    uint32_t addrVec;
    uint32_t dat;
    uint32_t en;
  };

  struct ExchangeReadMetaInfo {
    uint32_t exchangeVal;
    uint32_t localVal;
  };

  struct ExchangeWriteMetaInfo {
    uint32_t localVal;
    uint32_t exchangeVal;
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
        default: {
          llvm::dbgs() << "Error: Unknow codegen op name " << static_cast<uint32_t>(opName) << "\n";
          llvm_unreachable("Unsupported op");
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
        default: {
          llvm::dbgs() << "Error: Unknow codegen op name " << static_cast<uint32_t>(opName) << "\n";
          llvm_unreachable("Unsupported op");
        }
      }
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
    bool isIO;
  };

  struct CGMemMetaInfo {
    std::optional<mlir::StringRef> namehint;
    uint32_t bitWidth;
    uint64_t memDepth;
    uint32_t fragment_id;
    bool hasMultipleWriter;
    uint64_t memBase;
  };



  struct CGExchangeValueMetaInfo {
    // value is valid if isConst
    uint32_t writerId;
    mlir::SmallVector<uint32_t> readerIds;
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
    // Exchange values
    mlir::SmallVector<CGExchangeValueMetaInfo> exchangePool;

    mlir::SmallVector<uint32_t> r1Partitions, r2Partitions;

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

    mlir::DenseSet<mlir::StringRef> ioSignals;
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
    mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regs;

    // mlir::SmallVector<toucan::DefConstVectorOp*> constVecOps;
    
    DesignGraph(mlir::Operation *op, mlir::AnalysisManager &am);

    static bool opShouldRemoveInGraph(mlir::Operation *op);
  private:

    // TODO: Move levelize to here
  
  };


  class SchedulerBase {
  public:
    void collectPrintString(DesignGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t>  &printStrings);
  };


  class SingleRegionScheduler: SchedulerBase {
  public:
    mlir::SmallVector<mlir::BitVector> partitions;
    mlir::SmallVector<uint32_t> vtxIdToPartId;
    // partition i -> level j -> elem k
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> partLevels;

    CGInfo codeGenInfo;

    void levelizePartitions(DesignGraph &graph);

    void schedule(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);

    void fillDebugInfo();

  private:

    void collectConstant(DesignGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId);
    // void collectPrintMsgs(DesignGraph &graph);
    void generateRegMemLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);

  };



  class TwoRegionScheduler: SchedulerBase {
  public:
    PartitioningGraph r1Graph;
    PartitioningGraph r2Graph;
    mlir::SmallVector<mlir::BitVector> r1Partitions;
    mlir::SmallVector<mlir::BitVector> r2Partitions;
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> r1PartLevels;
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> r2PartLevels;

    mlir::SmallVector<mlir::SmallVector<uint32_t>> graphLevels;

    CGInfo codeGenInfo;

    void levelizeGraph(DesignGraph &graph);
    uint32_t findCutPoint(DesignGraph &graph, float r1Weight = 0.5);
    void cutGraph(DesignGraph &graph, uint32_t cutLevel = 10);
    // void generateCutGraph(uint32_t cutPoint);
    // void splitTwoRegion();
  
  private:


  };


  class NaivePartitioner: public SingleRegionScheduler {
    public:
    // NaivePartitioner();
    void partitionAndSchedule(DesignGraph &graph);
  };

  class RepCutPartitioner: public TwoRegionScheduler {
    public:
    RepCutPartitioner(std::filesystem::path outputDirectory) : outputDirectory(outputDirectory) {};
    void partitionAndSchedule(DesignGraph &graph);

    private:
    std::filesystem::path outputDirectory;
    
    void dumpGraphToFile(const PartitioningGraph &g, std::string fileName) const;
  };

}
