
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
#include "toucan/PartitioningGraph.h"

#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <filesystem>

namespace toucan {



  struct CGValueMetaInfo {
    bool isConst;
    bool isPlaceholder;
    // value is valid if isConst
    uint8_t value;
    mlir::Operation *definingOp;

    uint32_t levelId;
    // uint32_t opId;
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

  struct CGExchangeReadMetaInfo {
    uint32_t exchangeVal;
    uint32_t localVal;
  };

  struct CGExchangeWriteMetaInfo {
    uint32_t localVal;
    uint32_t exchangeVal;
  };


  struct CGOpMetaInfo {
    CGToucanOPName opName;
    
    union {
      // top level
      CGMemReadOpMetaInfo memRead;
      CGRegReadOpMetaInfo regRead;
      CGExchangeReadMetaInfo exgRead;

      // middle levels, exec
      CGLUTOpMetaInfo lut;
      CGVectorReadOpMetaInfo vec;

      // last level
      CGMemWriteOpMetaInfo memWrite;
      CGRegWriteOpMetaInfo regWrite;
      CGPrintOpMetaInfo print;
      CGStopOpMetaInfo stop;
      CGExchangeWriteMetaInfo exgWrite;
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
        case CGToucanOPName::ExchangeRead: {
          exgRead.localVal = result;
          return;
        }
        case CGToucanOPName::ExchangeWrite: {
          exgWrite.exchangeVal = result;
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
      llvm::dbgs() << "Error: Unknow codegen op name " << static_cast<uint32_t>(opName) << "\n";
      llvm_unreachable("Should not reach here");
    }
    uint32_t getResult() {
      switch (opName) {
        case CGToucanOPName::LUT: return lut.result;
        case CGToucanOPName::VecRead: return vec.result;
        case CGToucanOPName::RegRead: return regRead.result;
        case CGToucanOPName::MemRead: return memRead.result;
        case CGToucanOPName::ExchangeRead: return exgRead.localVal;
        case CGToucanOPName::ExchangeWrite: return exgWrite.exchangeVal;

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
      llvm::dbgs() << "Error: Unknow codegen op name " << static_cast<uint32_t>(opName) << "\n";
      llvm_unreachable("Unsupported op");
    }
    bool hasResult() {
      switch (opName) {
        case CGToucanOPName::LUT:
        case CGToucanOPName::VecRead:
        case CGToucanOPName::RegRead:
        case CGToucanOPName::MemRead: 
        case CGToucanOPName::ExchangeRead:
        case CGToucanOPName::ExchangeWrite:
          return true;

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
    bool isPadding;
    // write id (new)
    uint32_t writerId;
    uint32_t writerRegionId;
    mlir::Value val;
    // region, vtxId (new)
    mlir::SmallVector<std::tuple<uint32_t, uint32_t>> readerIds;
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
    uint32_t numExchangeReads;
    uint32_t numExchangeWrites;
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
    uint32_t numExchangeReads;
    uint32_t numExchangeWrites;

    void print() const;
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

    mlir::SmallVector<mlir::SmallVector<uint32_t>> regionPartitionIds;

    uint64_t totalMemSize;

    mlir::DenseMap<mlir::StringRef, uint32_t> printStrings;

    // debug info
    // name -> (fragment 0, 1, 2, ...)
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<uint32_t>> regDebugInfo;
    // name -> ((part, valId), (part, valId), ..)
    mlir::DenseMap<mlir::StringRef, mlir::SmallVector<std::tuple<uint32_t, uint32_t>>> signalDebugInfo;
    // name -> vector of all mem ids with same namehint. Start from fragment 0 to n
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
  
  };


  class SchedulerBase {
  public:
    CGInfo codeGenInfo;

    void fillDebugInfo();

    static void getVtxToLevel(const PartitioningGraph &g, mlir::SmallVector<uint32_t> &levels, uint32_t maxVtxId);
    static void collectPrintString(DesignGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t>  &printStrings);
    static void levelizeWorker(const PartitioningGraph &g, mlir::SmallVector<mlir::SmallVector<uint32_t>> &graphLevels);
    static void populateOpMetaDebugInfo(CGOpMetaInfo &opMeta, mlir::Operation *op);
  };


  class SingleRegionScheduler: public SchedulerBase {
  public:
    mlir::SmallVector<mlir::BitVector> partitions;
    mlir::SmallVector<uint32_t> vtxIdToPartId;
    // partition i -> level j -> elem k
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> partLevels;


    void levelizePartitions(DesignGraph &graph);

    void schedule(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);

    void fillDebugInfo();

  private:

    void collectConstant(DesignGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId);
    void generateRegMemLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);

  };



  class MultiRegionScheduler: public SchedulerBase {
    public:
    mlir::SmallVector<float> cutRatios;

    mlir::SmallVector<PartitioningGraph> regionGraphs;
    // regionId -> partitionId -> vtxes
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> regionPartitions;
    // regionId -> partitionId -> levelId -> vtxes
    mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>>> regionPartLevels;

    mlir::SmallVector<mlir::SmallVector<uint32_t>> graphLevels;

    mlir::SmallVector<uint32_t> cutPoints;

    // vtx id in graph -> vtx id in region graph
    // Map between old vertex to vertex in every region.
    // A node can be in only 1 region, thus mixing them together is fine.
    mlir::SmallVector<uint32_t> vtxIdToNewId;
    mlir::SmallVector<uint32_t> vtxIdToRegionId;
    // region -> newIdToOldId
    // UINT32_MAX means no corresponding vtx id in old graph
    mlir::SmallVector<mlir::SmallVector<uint32_t>> regionNewIdToVtxId;
    mlir::SmallVector<mlir::SmallVector<uint32_t>> regionNewIdToPartId;


    // CGInfo codeGenInfo;

    // GPU cache line size
    uint32_t partitionPaddingSpace = 128;
    uint32_t memPaddingSpace = 128;
    // if a memory has multiple writer, add extra padding to avoid possible write conflict. 
    // 4 => each memory element (4bits) takes 32 bits (4B)
    // Warning: Change this number also requires change in CodeGen and simulator!!
    uint32_t multiWriterMemElemBytes = 4;

    void levelizeGraphForCut(DesignGraph &graph);
    void findCutPoints(DesignGraph &graph);
    void cutGraph(DesignGraph &graph);

    //
    mlir::LogicalResult levelizeAllPartitions(mlir::MLIRContext *context);
    void schedule(DesignGraph &graph);


    private:

    // void collectConstant(DesignGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId);
    void sortRegistersForLocality(const PartitioningGraph &graph,  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered);
    void sortOpsAndExchangeValsForLocality(const mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regPoolOrdered, mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> &exchangeValIdOrdered);
    void generateRegMemLayout(DesignGraph &graph);
    void collectConstantVars(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> firstLevelOps);
    void collectConstantVecs(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId);

    // TODO: Share those infrastructure with SingleRegionScheduler
    static void scheduleFirstLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &firstLevelOps);
    void scheduleMiddleLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &currentLevel, uint32_t levelId);
    static void scheduleLastLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &lastLevel);

    static void scheduleExchangeReads(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &firstLevel);
    static void scheduleExchangeWrites(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &lastLevel);
  };


  class NaivePartitioner: public SingleRegionScheduler {
    public:
    // NaivePartitioner();
    mlir::LogicalResult partitionAndSchedule(DesignGraph &graph);
  };



  class RepCutPartitioningStatistics {
    public:
    mlir::SmallVector<uint32_t> partSize;
    mlir::SmallVector<uint32_t> partWeight;
    uint32_t graphSize, graphWeight;
    float sizeReplicationRate, weightReplicationRate, sizeIBFactor, weightIBFactor;
  };

  class RepCutPartitioner: public MultiRegionScheduler {
    public:
    float targetIb = 0.06;
    uint32_t rePartitionMaxIterations = 3;

    const uint32_t PARTITION_MAX_WEIGHT = 50000;
    const uint32_t REPARTITION_PREFERRED_WEIGHT = 50000;

    mlir::SmallVector<uint32_t> regionPartitionNumbers;

    RepCutPartitioner(std::filesystem::path outputDirectory) : outputDirectory(outputDirectory) {
      wholeGraphPath = outputDirectory / "design_before_cut.graph";
    };
    mlir::LogicalResult partitionAndSchedule(mlir::MLIRContext *context, DesignGraph &graph);

    void setPartitionTarget(uint32_t numRegions, uint32_t numPartsInEachRegion);

    private:
    std::filesystem::path outputDirectory;
    mlir::SmallVector<std::filesystem::path> regionWorkDirectory;
    std::filesystem::path wholeGraphPath;

    const char* graphFileName = "region.graph";
    const char* repcutOutputFileName = "rcp_output.txt";
    const char* repcutConsoleLogFileName = "repcut_print.txt";

    // float cutR1Weight = 0.4;
    
    void dumpGraphToFile(const PartitioningGraph &g, std::string fileName) const;

    mlir::LogicalResult callRepCutAndWait(uint32_t nParts, float target_ib, const std::string &graphFile, const std::filesystem::path &workingDirectory);

    mlir::LogicalResult parseRepCutResult(uint32_t nParts, const std::string &resultFile, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partitions);

    RepCutPartitioningStatistics getPartitionStatistics(uint32_t regionId);
    void printPartitionStatistics(const RepCutPartitioningStatistics &stats);

    mlir::LogicalResult workerFunc(const PartitioningGraph &graph, std::filesystem::path workDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, uint32_t nParts);

    mlir::LogicalResult rePartition(uint32_t regionId, const PartitioningGraph &graph, std::filesystem::path regionWorkDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput);
  };

}
