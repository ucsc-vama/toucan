
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
#include <optional>

#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "toucan/ToucanCodeGenInfo.h"

#include "toucan/MicroPartitioner.h"

#include <boost/graph/adjacency_list.hpp>

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


  class SchedulerBase {
  public:

    // void fillDebugInfo(bool fillSignalDebugInfo = true);

    static void getVtxToLevel(const PartitioningGraph &g, mlir::SmallVector<uint32_t> &levels, uint32_t maxVtxId);
    static void collectPrintString(const DesignGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t>  &printStrings);
    static void collectPrintString(const PartitioningGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t>  &printStrings);
    static void levelizeWorker(const PartitioningGraph &g, mlir::SmallVector<mlir::SmallVector<uint32_t>> &graphLevels);
    static void populateOpMetaDebugInfo(CGOpMetaInfo &opMeta, mlir::Operation *op);
  };


  class SingleRegionScheduler: public SchedulerBase {
  public:
    CGInfo codeGenInfo;

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


  class SingleRegionMicroPartScheduler: SchedulerBase {
    public:
    CGInfo codeGenInfo;

    std::vector<toucan::MicroPartitioner> mpartitioners;
    mlir::SmallVector<mlir::SmallVector<uint32_t>> repcutPartitions;



    void schedule(mlir::MLIRContext *context, const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeLis);
    void generateRegMemLayout(const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeList);
    

    private:
    // GPU cache line size
    const uint32_t partitionAlignment = 128;
    // uint32_t partitionPaddingSpace = 128;
    const uint32_t memPaddingSpace = 128;
    // if a memory has multiple writer, add extra padding to avoid possible write conflict. 
    // 4 => each memory element (4bits) takes 32 bits (4B)
    // Warning: Change this number also requires change in CodeGen and simulator!!
    const uint32_t multiWriterMemElemBytes = 4;

    mlir::DenseMap<mlir::Value, uint32_t> constValToRawValue;


    // procedures of generateRegMemLayout
    void sortRegistersForLocality(const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeList, mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered);
    void fillRegPool(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered);
    void fillMemPool(const PartitioningGraph &graph);
    void sortRegWriteOps(const PartitioningGraph &graph, mlir::SmallVector<uint32_t> &allRegWrites) const;
    void sortRegReadOps(const PartitioningGraph &graph, mlir::SmallVector<uint32_t> &allRegReads) const;

    void collectConstantVars(const PartitioningGraph &graph);
    void collectConstantVecs(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &partNodes);

    void scheduleRegReads(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allRegReads);
    
    void scheduleRegWrites(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allRegWrites);
    void scheduleMemWrites(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allMemWrites);
    void scheduleStops(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allStops);
    void schedulePrints(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allPrints);

    void scheduleMicroParts(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const std::vector<std::vector<MicroPart>> &partLevels);


    // First, place all regs and mems
    // Second, place values in each partition
    // Last, schedule ops inside each partition
  };

  class MultiRegionPartitioner: SchedulerBase {
    public:
    CGInfo codeGenInfo;
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



    // // GPU cache line size
    // const uint32_t partitionAlignment = 128;
    // // uint32_t partitionPaddingSpace = 128;
    // uint32_t memPaddingSpace = 128;
    // // if a memory has multiple writer, add extra padding to avoid possible write conflict. 
    // // 4 => each memory element (4bits) takes 32 bits (4B)
    // // Warning: Change this number also requires change in CodeGen and simulator!!
    // uint32_t multiWriterMemElemBytes = 4;

    void levelizeGraphForCut(DesignGraph &graph);
    void findCutPoints(DesignGraph &graph);
    void cutGraph(DesignGraph &graph);
    void breakDirectIOConnection();

    void doNotCutGraph(DesignGraph &graph);

    mlir::LogicalResult levelizeAllPartitions(mlir::MLIRContext *context);
    // void schedule(DesignGraph &graph);


    private:

    mlir::DenseMap<mlir::Operation*, uint32_t> vecDeclMovedToLaterRegion;
    mlir::DenseMap<mlir::Value, uint32_t> constValToRawValue;

    // void sortRegistersForLocality(const PartitioningGraph &graph,  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered);
    // void sortRegWriteOps(mlir::DenseMap<mlir::Value, uint32_t> &regValToOrder);
    // void sortRegReadOps(mlir::DenseMap<mlir::Value, uint32_t> &regValToOrder);
    // void generateRegMemLayout(DesignGraph &graph);


    // void groupExchangeVals(mlir::SmallVector<mlir::SmallVector<uint32_t>> &exchangeValIdOrdered);
    // void sortExchangeWriteOps();
    // void sortExchangeReadOps();
    // void generateExchangeLayout();


    // void collectConstantVars(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> firstLevelOps);
    // void collectConstantVecs(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId);


    // void scheduleRegReads(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &firstLevelOps);
    // void scheduleMiddleLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &currentLevel, uint32_t levelId);
    // void scheduleLastLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &lastLevel);

    // void scheduleExchangeReads(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &firstLevel);
    // void scheduleExchangeWrites(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &lastLevel);
  };


  // class NaivePartitioner: public SingleRegionScheduler {
  //   public:
  //   // NaivePartitioner();
  //   mlir::LogicalResult partitionAndSchedule(DesignGraph &graph);
  // };



  class RepCutPartitioningStatistics {
    public:
    mlir::SmallVector<uint32_t> partSize;
    mlir::SmallVector<uint32_t> partWeight;
    uint32_t graphSize, graphWeight;
    float sizeReplicationRate, weightReplicationRate, sizeIBFactor, weightIBFactor;
  };

  class RepCutPartitioner: public MultiRegionPartitioner {
    public:
    float targetIb = 0.015;
    uint32_t rePartitionMaxIterations = 4;

    const uint32_t PARTITION_MAX_WEIGHT = 50000;
    const uint32_t PARTITION_PREFERRED_WEIGHT = 45000;
    const uint32_t REPARTITION_PREFERRED_WEIGHT = 40000;

    mlir::SmallVector<uint32_t> regionPartitionNumbers;

    // VecDecl -> [vector element ids]
    mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> originalVectorElementsMap;

    RepCutPartitioner(std::filesystem::path outputDirectory) : outputDirectory(outputDirectory) {
      wholeGraphPath = outputDirectory / "design_before_cut.graph";
    };
    mlir::LogicalResult _partition(mlir::MLIRContext *context, DesignGraph &graph);
    mlir::LogicalResult _schedule(mlir::MLIRContext *context, DesignGraph &graph);
    
    void dumpAllPartitionsToFile();

    mlir::LogicalResult partitionAndSchedule(mlir::MLIRContext *context, DesignGraph &graph);

    void setPartitionTarget();

    // private:
    std::filesystem::path outputDirectory;
    mlir::SmallVector<std::filesystem::path> regionWorkDirectory;
    std::filesystem::path wholeGraphPath;

    const char* graphFileName = "region.graph";
    const char* graphVectorDeclInfoFileName = "vec_decl_info.txt";
    const char* repcutOutputFileName = "rcp_output.txt";
    const char* repcutConsoleLogFileName = "repcut_print.txt";

    private:
    void dumpGraphToFile(const PartitioningGraph &g, std::string fileName) const;
    void collectAndDumpGraphVectorDeclInfoToFile(const PartitioningGraph &g, std::string fileName);
    void dumpSinglePartitionToFile(const PartitioningGraph &g, mlir::SmallVector<uint32_t> partNodes, std::string fileName) const;

    mlir::LogicalResult callRepCutAndWait(uint32_t nParts, float target_ib, const std::string &graphFile, const std::filesystem::path &workingDirectory, int maxThreads);

    mlir::LogicalResult parseRepCutResult(uint32_t nParts, const std::string &resultFile, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partitions);

    RepCutPartitioningStatistics getPartitionStatistics(uint32_t regionId);
    void printPartitionStatistics(const RepCutPartitioningStatistics &stats);

    mlir::LogicalResult workerFunc(const PartitioningGraph &graph, std::filesystem::path workDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, uint32_t nParts, int maxThreads);

    mlir::LogicalResult rePartition(mlir::MLIRContext *context, uint32_t regionId, const PartitioningGraph &graph, std::filesystem::path regionWorkDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput);

    int decideRepCutNumThreads(int maxThreads, int numTargetPartitions);
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
