
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

#include "toucan/MicroPartitioner.h"

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
    mlir::SmallVector<uint32_t> nodes;
    // NodeId to ops * in this part *
    mlir::DenseMap<uint32_t, uint32_t> nodeToOpCount;
    // LUT(and VecDecl), VecRead, RegRead, 
    CGToucanOPName opType;
    uint32_t totalOpCount;


    mlir::SmallVector<mlir::SmallVector<uint32_t>> levels;
    mlir::DenseSet<mlir::Value> inputValues, outputValues;
    int max_live_vars, num_input_vars, num_output_vars;

    mlir::LogicalResult check();
    void schedule();
    void clear();

    void buildRegularLUTPart(const mlir::SmallVector<uint32_t> &newNodes);
    void buildSpecialPart(const CGToucanOPName vtxOpName, const mlir::SmallVector<NodeIdAndOpCount> &nodeAllocation);
  };

  // A secondary level partitioner. Accepts repcut graph file.
  class MicroPartitioner {
    public:
    std::vector<std::vector<MicroPart>> partLevels;
    // RegReads at first level, should be ordered by its memory location in the scheduler
    mlir::SmallVector<uint32_t> allRegReads;
    // RegWrite, MemWrite, ExchangeWrite should be moved to last
    mlir::SmallVector<uint32_t> allRegWrites, allMemWrites;
    // Stop, Print can be moved to last
    mlir::SmallVector<uint32_t> allStops, allPrints;

    MicroPartitioner(std::filesystem::path workDirectory, size_t partId) : partId(partId), workDirectory(workDirectory) {
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
    
      graphVectorInfoFile = workDirectory / graphVectorDeclInfoFileName;
    };

    mlir::LogicalResult partition();
    mlir::LogicalResult arrangeSpecialOps(PartitioningGraph &g);


    // TODO: Move those 3 to Scheduler
    mlir::LogicalResult scheduleNormalMicroParts(PartitioningGraph &g);
    mlir::LogicalResult scheduleSpecialMicroParts(PartitioningGraph &g);
    // Note: This must be done after MicroPartLocalValAllocator
    mlir::LogicalResult scheduleIOMicroParts(PartitioningGraph &g);


    private:
    size_t partId;
    std::filesystem::path workDirectory;
    std::string inputGraphFile, consoleLogFile, outputFile, graphVectorInfoFile;

    const char* graphFilePrefix = "dump_part_";
    const char* graphFileSuffix = ".graph";
    const char* consoleLogFilePrefix = "micro_part_print_";
    const char* consoleLogFileSuffix = ".log";
    const char* outputFilePrefix = "micro_part_result_";
    const char* outputFileSuffix = ".txt";
    const char* graphVectorDeclInfoFileName = "vec_decl_info.txt";
    const char* pythonName = "python3";


    mlir::SmallVector<mlir::SmallVector<uint32_t>> excludeNodeLevels;

    // TODO: Update this path!!!!!
    const char* microPartitionerPythonPath = "/Users/hwang/project/gsim/graph-dse/MicroPartitioner.py";

    
    mlir::LogicalResult callExternalPartitioner();
    mlir::LogicalResult loadMicroParts();

    // TODO: reorganize exclude nodes

  };

}