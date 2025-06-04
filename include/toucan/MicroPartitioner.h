
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

#define DESIGNGRAPH_EXGREAD_WEIGHT 1
#define DESIGNGRAPH_EXGWRITE_WEIGHT 1
#define DESIGNGRAPH_BREAK_IO_NOP_WEIGHT 0

#define GPU_THREAD_WARP_SIZE 32

namespace toucan {

  struct NodeIdAndOpCount {
    uint32_t node;
    uint32_t opCount;
  };

  // MicroPart data structure
  class MicroPart {
    public:
    uint32_t lineno;

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

    // Valid for special ops
    mlir::SmallVector<mlir::Operation*> specialOps;



    // void schedule();
    void clear();

    void buildRegularLUTPart(const mlir::SmallVector<mlir::SmallVector<uint32_t>> &newNodesLevel);
    void buildSpecialPart(const CGToucanOPName vtxOpName, const mlir::SmallVector<mlir::Operation*> &rawOps);


    
    bool checkAndCollectIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId);

    // void moveOutputValsToLastLevel();
    // void collectValueLifetime();
    // void addDummyNOPs();
    // // void 

    // // TODO
    // void sortOpsForLocality();

    private:
    void updateNodeToLevel();

    bool checkAndCollectRegularPartIOValues(const PartitioningGraph &g, const mlir::DenseSet<uint32_t> &allNodes, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId);

    bool checkAndCollectSpecialPartIOValues(const PartitioningGraph &g, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToDepNodeId, const mlir::DenseMap<uint32_t, uint32_t> &newNodeIdToOriginalVecDeclId);
  };

  // A secondary level partitioner. Accepts repcut graph file.
  class MicroPartitioner {
    public:
    mlir::DenseSet<uint32_t> allNodes;
    std::vector<std::vector<MicroPart>> partLevels;
    // RegReads at first level, should be ordered by its memory location in the scheduler
    mlir::SmallVector<uint32_t> allRegReads;
    // RegWrite, MemWrite, ExchangeWrite should be moved to last
    mlir::SmallVector<uint32_t> allRegWrites, allMemWrites;
    // Stop, Print can be moved to last
    mlir::SmallVector<uint32_t> allStops, allPrints;

    mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> outputVectorNopMap;
    // VecDecl -> [vector element ids]
    mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> originalVectorElementsMap;
    mlir::DenseMap<uint32_t, uint32_t> newNodeIdToOriginalVecDeclId;
    mlir::DenseMap<uint32_t, uint32_t> newNodeIdToDepNodeId;

    MicroPartitioner(const mlir::SmallVector<uint32_t> &thisRepCutPartition, std::filesystem::path workDirectory, size_t partId, mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> vectorElementsMap) : originalVectorElementsMap(vectorElementsMap), partId(partId), workDirectory(workDirectory) {
      allNodes.insert(thisRepCutPartition.begin(), thisRepCutPartition.end());
      assert(allNodes.size() == thisRepCutPartition.size() && "RepCut partition should not have duplicated node!");

      std::ostringstream oss;
      oss << graphFilePrefix << partId << graphFileSuffix;
      inputGraphFile = workDirectory / oss.str();
    
      oss.clear();
      oss.str("");
      oss << consoleLogFilePrefix << partId << consoleLogFileSuffix;
      consoleLogFile = workDirectory / oss.str();
      
      oss.clear();
      oss.str("");
      oss << outputFilePrefix << partId << outputFileSuffix;
      outputFile = workDirectory / oss.str();

      oss.clear();
      oss.str("");
      oss << outputVectorMapFilePrefix << partId << outputVectorMapFileSuffix;
      outputVectorMapFile = workDirectory / oss.str();
    
      graphVectorInfoFile = workDirectory / graphVectorDeclInfoFileName;
    };

    mlir::LogicalResult partition();
    mlir::LogicalResult arrangeSpecialOps(PartitioningGraph &g);

    void collectPartIOValues(const PartitioningGraph &g);


    private:
    size_t partId;
    std::filesystem::path workDirectory;
    std::string inputGraphFile, consoleLogFile, outputFile, outputVectorMapFile, graphVectorInfoFile;

    const char* graphFilePrefix = "dump_part_";
    const char* graphFileSuffix = ".graph";
    const char* consoleLogFilePrefix = "micro_part_print_";
    const char* consoleLogFileSuffix = ".log";
    const char* outputFilePrefix = "micro_part_result_";
    const char* outputFileSuffix = ".txt";
    const char* graphVectorDeclInfoFileName = "vec_decl_info.txt";
    const char* outputVectorMapFilePrefix = "micro_part_result_vecmap_";
    const char* outputVectorMapFileSuffix = ".txt";
    const char* pythonName = "python3";


    mlir::SmallVector<mlir::SmallVector<uint32_t>> excludeNodeLevels;

    // TODO: Update this path!!!!!
    const char* microPartitionerPythonPath = "/Users/hwang/project/gsim/graph-dse/MicroPartitioner.py";

    
    mlir::LogicalResult callExternalPartitioner();
    mlir::LogicalResult loadMicroParts();
    mlir::LogicalResult loadVectorNopMap();

    // TODO: reorganize exclude nodes

  };

}