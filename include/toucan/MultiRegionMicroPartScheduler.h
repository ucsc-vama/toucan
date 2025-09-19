
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
#include "toucan/SchedulerBase.h"

#include "toucan/MicroPartitioner.h"

#include <boost/graph/adjacency_list.hpp>

#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>

#include "toucan/PartitioningManager.h"


#define GPU_THREAD_WARP_SIZE 32

namespace toucan {


  // class SingleRegionScheduler: public SchedulerBase {
  // public:
  //   CGInfo codeGenInfo;

  //   mlir::SmallVector<mlir::BitVector> partitions;
  //   mlir::SmallVector<uint32_t> vtxIdToPartId;
  //   // partition i -> level j -> elem k
  //   mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> partLevels;


  //   void levelizePartitions(DesignGraph &graph);

  //   void schedule(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);

  //   void fillDebugInfo();

  // private:

  //   void collectConstant(DesignGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId);
  //   void generateRegMemLayout(DesignGraph &graph, uint32_t partitionRegPaddingSpace = 32, uint32_t memPaddingSpace = 32);

  // };


  // class SingleRegionMicroPartScheduler: SchedulerBase {
  //   public:
  //   CGInfo codeGenInfo;

  //   std::vector<toucan::MicroPartitioner> mpartitioners;
  //   mlir::SmallVector<mlir::SmallVector<uint32_t>> repcutPartitions;



  //   void schedule(mlir::MLIRContext *context, const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeLis);
  //   void generateRegMemLayout(const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeList);
    

  //   private:
  //   // GPU cache line size
  //   const uint32_t partitionAlignment = 128;
  //   // uint32_t partitionPaddingSpace = 128;
  //   const uint32_t memPaddingSpace = 128;
  //   // if a memory has multiple writer, add extra padding to avoid possible write conflict. 
  //   // 4 => each memory element (4bits) takes 32 bits (4B)
  //   // Warning: Change this number also requires change in CodeGen and simulator!!
  //   const uint32_t multiWriterMemElemBytes = 4;

  //   mlir::DenseMap<mlir::Value, uint32_t> constValToRawValue;


  //   // procedures of generateRegMemLayout
  //   void sortRegistersForLocality(const PartitioningGraph &graph, const mlir::SmallVector<mlir::SmallVector<uint32_t>> &partNodeList, mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered);
  //   void fillRegPool(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered);
  //   void fillMemPool(const PartitioningGraph &graph);
  //   void sortRegWriteOps(const PartitioningGraph &graph, mlir::SmallVector<uint32_t> &allRegWrites) const;
  //   void sortRegReadOps(const PartitioningGraph &graph, mlir::SmallVector<uint32_t> &allRegReads) const;

  //   void collectConstantVars(const PartitioningGraph &graph);
  //   void collectConstantVecs(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &partNodes);

  //   void scheduleRegReads(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allRegReads);
    
  //   void scheduleRegWrites(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allRegWrites);
  //   void scheduleMemWrites(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allMemWrites);
  //   void scheduleStops(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allStops);
  //   void schedulePrints(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> &allPrints);

  //   void scheduleMicroParts(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const std::vector<std::vector<MicroPart>> &partLevels);

  //   std::mutex debugSymbolLock;

  //   void fillSignalDebugInfoForSinglePart(const MicroPartLocalValueAllocator &valAllocator, uint32_t partId);
  //   void fillDebugInfo();


  //   // First, place all regs and mems
  //   // Second, place values in each partition
  //   // Last, schedule ops inside each partition
  // };


  class MultiRegionMicroPartScheduler: public SchedulerBase {
    public:
    CGInfo codeGenInfo;

    std::vector<std::vector<RepCutPartitionCodeGenData>> regionPartData;



    void schedule(mlir::MLIRContext *context, const PartitioningGraph &rawGraph, const mlir::SmallVector<mlir::Value> &exchangeValPool);
    void buildDummyVtxIndexInVec(const MicroPartitioner mPartitioner);
    void copyVecTables(const MicroPartitioner mPartitioner);



    

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
    // mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> regToOrder;
    // mlir::DenseMap<mlir::Value, uint32_t> exchangeValToOrder;

    // need build
    mlir::DenseMap<uint32_t, uint32_t> dummyVtxIndexInVecTable;
    // copy from micro partitioner
    // vecDecl -> [vec nop (new)]
    mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> outputVectorNopMap;
    // VecDecl -> [vector element ids (old)]
    // mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> originalVectorElementsMap;
    mlir::DenseMap<uint32_t, uint32_t> newNodeIdToOriginalVecDeclId;
    mlir::DenseMap<uint32_t, uint32_t> newNodeIdToDepNodeId;


    // procedures of generateRegMemLayout
    void sortRegistersForLocality(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered);
    void sortRegistersForLocality_2(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered);
    void fillRegPool(mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered);
    // void fillRegOrderTable();
    void fillMemPool(const PartitioningGraph &graph);
    void sortRegWriteOps(RepCutPartitionCodeGenData &partMeta) const;
    void sortRegReadOps(RepCutPartitionCodeGenData &partMeta) const;

    // procedures of generateExchangeLayout
    void sortExchangeValsForLocality(mlir::SmallVector<mlir::SmallVector<mlir::Value>> &exgValOrdered);
    void fillExchangePool(mlir::SmallVector<mlir::SmallVector<mlir::Value>> &exgValOrdered);
    // void fillExgOrderTable();
    void sortExchangeReadOps(RepCutPartitionCodeGenData &partData) const;
    void sortExchangeWriteOps(RepCutPartitionCodeGenData &partData) const;

    void collectConstantVars(const PartitioningGraph &graph);
    void collectConstantVecs(const RepCutPartitionCodeGenData &partData, CGPartitionMetaInfo &partInfo);

    void generateRegMemLayout(const PartitioningGraph &rawGraph);
    void generateExchangeLayout(const mlir::SmallVector<mlir::Value> &exchangeValPool);

  

    void scheduleRegReads(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::RegReadOp> &allRegReads);
    
    void scheduleRegWrites(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::RegWriteOp> &allRegWrites);
    void scheduleMemWrites(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::MemWriteOp> &allMemWrites);
    void scheduleStops(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::StopOp> &allStops);
    void schedulePrints(CGPartitionMetaInfo &partInfo, mlir::SmallVector<toucan::PrintOp> &allPrints);

    void scheduleExchangeReads(CGPartitionMetaInfo &partInfo, const mlir::SmallVector<mlir::Value> &allExgReadVals);
    void scheduleExchangeWrites(CGPartitionMetaInfo &partInfo, const mlir::SmallVector<mlir::Value> &allExgWriteVals);


    void scheduleRegularMicroPart(const PartitioningGraph &graph, CGMicroPartInfo &part, const MicroPart &mPart, const mlir::DenseMap<mlir::Value, uint32_t> &valToValId) const;
    void scheduleSpecialMicroPart(const PartitioningGraph &graph, CGMicroPartInfo &part, const MicroPart &mPart, const CGPartitionMetaInfo &partInfo) const;
    void scheduleNOPMicroPart(CGMicroPartInfo &part, MicroPart &mPart, const mlir::DenseMap<mlir::Value, uint32_t> &valToValId) const;
    void scheduleMicroParts(const PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const std::vector<std::vector<MicroPart>> &partLevels);

    static void mergeSmallRegularMParts(std::vector<CGMicroPartInfo> &mPartsThisLevel);

    std::mutex debugSymbolLock;

    void fillSignalDebugInfoForSinglePart(const MicroPartLocalValueAllocator &valAllocator, uint32_t partId);
    void fillDebugInfo();

    static void printPartInfo(const CGPartitionMetaInfo &partInfo);

  };

}
