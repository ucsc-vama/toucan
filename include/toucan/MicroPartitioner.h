
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
#include "toucan/MicroPart.h"


#include <boost/graph/adjacency_list.hpp>

#include <unordered_map>
#include <filesystem>
#include <vector>

#define DESIGNGRAPH_EXGREAD_WEIGHT 1
#define DESIGNGRAPH_EXGWRITE_WEIGHT 1
#define DESIGNGRAPH_BREAK_IO_NOP_WEIGHT 0

#define GPU_THREAD_WARP_SIZE 32

namespace toucan {

  struct NodeIdAndOpCount {
    uint32_t node;
    uint32_t opCount;
  };


  // A secondary level partitioner. Accepts repcut graph file.
  class MicroPartitioner {
    public:
    mlir::DenseSet<uint32_t> allNodes;
    std::vector<std::vector<std::shared_ptr<MicroPart>>> partLevels;

    mlir::SmallVector<mlir::SmallVector<uint32_t>> excludeNodeLevels;
    // RegReads at first level, should be ordered by its memory location in the scheduler
    mlir::SmallVector<uint32_t> allRegReads;
    mlir::SmallVector<uint32_t> allExgReads;
    // RegWrite, MemWrite, ExchangeWrite should be moved to last
    mlir::SmallVector<uint32_t> allRegWrites;
    mlir::SmallVector<uint32_t> allMemWrites;
    mlir::SmallVector<uint32_t> allExgWrites;
    // Stop, Print can be moved to last
    mlir::SmallVector<uint32_t> allStops, allPrints;

    // vecDecl -> [vec nop (new)]
    mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> outputVectorNopMap;
    // VecDecl -> [vector element ids (old)]
    mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> originalVectorElementsMap;
    mlir::DenseMap<uint32_t, uint32_t> newNodeIdToOriginalVecDeclId;
    mlir::DenseMap<uint32_t, uint32_t> newNodeIdToDepNodeId;



    void init(const mlir::SmallVector<uint32_t> &thisRepCutPartition, const std::filesystem::path workDirectory, const size_t partId, const mlir::DenseMap<uint32_t, mlir::SmallVector<uint32_t>> vectorElementsMap) {
      originalVectorElementsMap = vectorElementsMap;
      this->partId = partId;
      this->workDirectory = workDirectory;

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
    mlir::LogicalResult arrangeSpecialOps(const PartitioningGraph &g, const size_t maxOpsPerMPart = 32);
    void mergeSpecialMParts(const size_t maxOpsPerMPart);
    void collectPartIOValues(mlir::MLIRContext *context, const PartitioningGraph &g);

    std::string inputGraphFile, consoleLogFile, outputFile, outputVectorMapFile, graphVectorInfoFile;

    private:
    size_t partId;
    std::filesystem::path workDirectory;

    const char* graphFilePrefix = "dump_part_";
    const char* graphFileSuffix = ".graph";
    const char* consoleLogFilePrefix = "micro_part_print_";
    const char* consoleLogFileSuffix = ".log";
    const char* outputFilePrefix = "micro_part_result_";
    const char* outputFileSuffix = ".txt";
    const char* graphVectorDeclInfoFileName = "vec_decl_info.txt";
    const char* outputVectorMapFilePrefix = "micro_part_result_vecmap_";
    const char* outputVectorMapFileSuffix = ".txt";

    const char* microPartitionerBin = "toucan-mpart";

    
    mlir::LogicalResult callExternalPartitioner();
    mlir::LogicalResult loadMicroParts();
    mlir::LogicalResult loadVectorNopMap();

  };

}