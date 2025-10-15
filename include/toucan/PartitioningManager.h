
#pragma once


#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"



#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"


#include <cstdint>


#include "toucan/ToucanConfigs.h"
#include "toucan/PartitioningGraph.h"

#include "toucan/MicroPartitioner.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"

#include <boost/graph/adjacency_list.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>



namespace toucan {

  class RepCutPartitionCodeGenData {
    public:

    mlir::SmallVector<mlir::SmallVector<std::shared_ptr<MicroPart>>> mpartLevels;

    mlir::SmallVector<toucan::RegReadOp> allRegReads;
    mlir::SmallVector<toucan::RegWriteOp> allRegWrites;
    mlir::SmallVector<toucan::PrintOp> allPrints;
    mlir::SmallVector<toucan::StopOp> allStops;
    mlir::SmallVector<toucan::MemWriteOp> allMemWrites;

    mlir::SmallVector<mlir::Value> allExgReadVals, allExgWriteVals;
    mlir::SmallVector<mlir::TypedValue<toucan::RegType>> readRegs, writeRegs;
  };
  
  class PartitioningManager {
    private:
    mlir::DenseMap<mlir::Value, mlir::DenseSet<mlir::Operation*>> valToUserOpsDoNotReplace;

    public:
    mlir::MLIRContext *context;

    std::filesystem::path outputDirectory;
    std::filesystem::path microPartitionerWorkDir;
    std::filesystem::path vectorElementsMapFilePath;
    mlir::SmallVector<std::filesystem::path> regionWorkDirectory;
    std::filesystem::path rawGraphPath;


    const char* graphFileName = "design.graph";
    const char* graphVectorDeclInfoFileName = "vec_decl_info.txt";
    const char* repcutOutputFileName = "rcp_output.txt";
    const char* repcutConsoleLogFileName = "repcut_print.txt";



    std::unique_ptr<MicroPartitioner> mp;

    std::unique_ptr<PartitioningGraph> microPartGraph;
    std::shared_ptr<PartitioningGraph> microPartGraph_r0;
    std::shared_ptr<PartitioningGraph> microPartGraph_r1;

    size_t mpGraphTotalWeight_r0;
    size_t mpGraphTotalWeight_r1;

    mlir::SmallVector<uint32_t> newNOPVtxes_r0;
    mlir::SmallVector<uint32_t> newNOPVtxes_r1;
    mlir::SmallVector<mlir::SmallVector<uint32_t>> repcutPartitions_r0;
    mlir::SmallVector<mlir::SmallVector<uint32_t>> repcutPartitions_r1;

    mlir::SmallVector<mlir::SmallVector<uint32_t>> microPartGraph_levels;
    mlir::SmallVector<mlir::SmallVector<uint32_t>> microPartGraph_r0_levels;
    mlir::SmallVector<mlir::SmallVector<uint32_t>> microPartGraph_r1_levels;

    mlir::DenseMap<mlir::Value, uint32_t> exchangeValues;
    mlir::SmallVector<mlir::Value> exchangeValPool;

    // output, use for code gen
    std::vector<RepCutPartitionCodeGenData> partCodeGenData_r0;
    std::vector<RepCutPartitionCodeGenData> partCodeGenData_r1;

    mlir::DenseMap<uint32_t, uint32_t> mpGraph_idToRawGraphId;

    // VecDecl -> [vector element ids]
    mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> rawGraphVectorElementsMap;

    mlir::LogicalResult init(std::filesystem::path outputDirectory, int numRegions);

    mlir::LogicalResult runStage1MicroPartitioner(const PartitioningGraph &rawGraph);

    mlir::LogicalResult buildMicroPartGraph(const PartitioningGraph &rawGraph);


    int findCutPoint();
    void cutGraph(int cutPoint);
    void breakDirectIOConnection(const PartitioningGraph &rawGraph);

    mlir::SmallVector<uint32_t> breakDirectIOConnectionWorker(PartitioningGraph &g, const mlir::DenseSet<mlir::Operation*> &rawOpsInFollowingRegions);


    void updateGraphWeight_r0();
    void updateGraphWeight_r1();
    mlir::LogicalResult runStage2RepCutPartitioner(float partSizeRatio, float ibFactor);

    void collectRepCutPartitionCodeGenData();


    static mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> collectGraphVectorDeclInfoToFile(const PartitioningGraph &rawGraph);

    static mlir::LogicalResult dumpGraphVectorDeclInfoToFile(std::string fileName, const mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> &originalVectorElementsMap);

    void levelizeGraph(const PartitioningGraph &g, mlir::SmallVector<mlir::SmallVector<uint32_t>> &graphLevels) const;




    // mlir::LogicalResult collectAndDumpGraphVectorDeclInfoToFile(const uint32_t regionId, const PartitioningGraph &g, std::string fileName);

    static mlir::LogicalResult dumpGraphToFileForMicroPartitioner(const PartitioningGraph &g, mlir::SmallVector<uint32_t> partNodes, std::string fileName);

    static mlir::LogicalResult dumpGraphToFileForRepCut(const PartitioningGraph &g, std::string fileName);

  };


}
