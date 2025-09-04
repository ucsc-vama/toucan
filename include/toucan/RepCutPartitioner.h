
#pragma once

#include "mlir/IR/MLIRContext.h"


#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"


#include <cstdint>
#include <memory>

#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanConfigs.h"


#include <boost/graph/adjacency_list.hpp>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>


namespace toucan {


  class RepCutPartitioningStatistics {
    public:
    mlir::SmallVector<uint32_t> partSize;
    mlir::SmallVector<uint32_t> partWeight;
    uint32_t graphSize, graphWeight;
    float sizeReplicationRate, weightReplicationRate, sizeIBFactor, weightIBFactor;
  };

  class RepCutPartitioner {
    public:
    float targetIb = 0.015;
    uint32_t rePartitionMaxIterations = 20;


    uint32_t PARTITION_MAX_WEIGHT = 1650000;
    uint32_t PARTITION_PREFERRED_WEIGHT = 1450000;
    float REPARTITION_SIZE_TARGET_RATIO = 1.3;


    // mlir::SmallVector<uint32_t> regionPartitionNumbers;
    uint32_t repcutTargetPartCount;
    // uint32_t repcutFinalPartCount;
    mlir::SmallVector<mlir::SmallVector<uint32_t>> repcutPartitions;

    RepCutPartitioner(std::filesystem::path outputDirectory, std::shared_ptr<PartitioningGraph> graph) : outputDirectory(outputDirectory), _graph(graph) {
      repcutTargetPartCount = 0;
    };
    mlir::LogicalResult _partition(mlir::MLIRContext *context);

    void setPartitionTarget(float partSizeRatio, int targetGPUSMCount);

    // private:
    std::filesystem::path outputDirectory;
    // std::filesystem::path wholeGraphPath;

    const char* graphFileName = "region.graph";
    const char* repcutOutputFileName = "rcp_output.txt";
    const char* repcutConsoleLogFileName = "repcut_print.txt";

    private:

    std::shared_ptr<PartitioningGraph> _graph;


    static mlir::LogicalResult dumpGraphToFile(const PartitioningGraph &g, std::string fileName);


    mlir::LogicalResult callRepCutAndWait(uint32_t nParts, float target_ib, const std::string &graphFile, const std::filesystem::path &workingDirectory, int maxThreads);

    static mlir::LogicalResult parseRepCutResult(uint32_t nParts, const std::string &resultFile, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partitions);

    RepCutPartitioningStatistics getPartitionStatistics();
    void printPartitionStatistics(const RepCutPartitioningStatistics &stats);

    mlir::LogicalResult workerFunc(const PartitioningGraph &graph, std::filesystem::path workDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, uint32_t nParts, int maxThreads);

    bool keepRepartitionMayUseless;
    mlir::LogicalResult rePartition(mlir::MLIRContext *context, const PartitioningGraph &graph, std::filesystem::path regionWorkDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, const uint32_t iterId, uint32_t &previousRePartitionInputNumParts);

    int decideRepCutNumThreads(int maxThreads, int numTargetPartitions);
  };


}
