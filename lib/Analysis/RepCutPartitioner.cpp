
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <chrono>

#include <thread>
#include <sstream>
#include <iomanip>

#include <numeric>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

static uint32_t getPartWeight(const mlir::SmallVector<uint32_t> &part, const PartitioningGraph &graph) {
  uint32_t weight = 0;
  for (const auto eachVtx: part) {
    weight += graph[eachVtx].weight;
  }
  return weight;
};

void RepCutPartitioner::setPartitionTarget(uint32_t numRegions, uint32_t numPartsInEachRegion) {
  assert(numRegions > 1);
  assert(numPartsInEachRegion > 0);

  regionPartitionNumbers.resize(numRegions, numPartsInEachRegion);

  switch (numRegions) {
    case 4: {
      cutRatios = {0.35, 0.18, 0.18};
      break;
    }
    default: {
      // Policy: evenly distribute region size
      cutRatios.resize(numRegions - 1, 1.0f / (numRegions));
    }
  }
  
  assert(cutRatios.size() == numRegions - 1);
}

LogicalResult RepCutPartitioner::partitionAndSchedule(mlir::MLIRContext *context, DesignGraph &graph) {

  // Levelize
  llvm::outs() << "====================Levelize And Cut====================\n";
  levelizeGraphForCut(graph);

  // Cut into 2 subgraph
  findCutPoints(graph);
  cutGraph(graph);
  breakDirectIOConnection(graph);
  // Clear unneeded data
  graphLevels.clear();


  // auto numVtxes = boost::num_vertices(graph.g);
  auto numRegions = regionGraphs.size();
  assert(regionPartitionNumbers.size() == numRegions);

  for (size_t rid = 0; rid < numRegions; rid++) {
    std::ostringstream oss;
    oss << "region" << rid;
    std::string dirName = oss.str();
    regionWorkDirectory.push_back(outputDirectory / dirName);
  }

  assert(numRegions == regionPartitionNumbers.size());
  regionPartitions.clear();
  regionPartitions.resize(numRegions);

  mlir::SmallVector<uint32_t> regionIds(numRegions);
  std::iota(regionIds.begin(), regionIds.end(), 0);

  // llvm::outs() << "Start partitioning\n";
  llvm::outs() << "====================Partitioning====================\n";

  // Save graph to file for debug purpose.
  std::thread bgDumpThread([&]() {
    auto start = std::chrono::high_resolution_clock::now();
    dumpGraphToFile(graph.g, wholeGraphPath);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::ostringstream msgOss;
    msgOss << "(Optionally) Dump the whole graph spend " << duration << "ms\n";
    std::string msg = msgOss.str();
    llvm::outs() << msg;
  });

  auto ret = mlir::failableParallelForEach(context, regionIds.begin(), regionIds.end(), [&](uint32_t regionId) {
    auto start = std::chrono::high_resolution_clock::now();

    auto ret = workerFunc(
      regionGraphs[regionId], 
      regionWorkDirectory[regionId], 
      regionPartitions[regionId], 
      regionPartitionNumbers[regionId]);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::ostringstream msgOss;
    msgOss << "Graph region " << regionId << " spend " << duration << "ms\n";
    llvm::outs() << msgOss.str();

    uint32_t rePartIterationCount = 0;
    bool converged = false;
    uint32_t rawPartitionCount = regionPartitions[regionId].size();
    while ((rePartIterationCount <= rePartitionMaxIterations) && (!converged)) {
      msgOss.str("");
      msgOss.clear();
      msgOss << "Repartition iteration " << rePartIterationCount << "\n";
      llvm::outs() << msgOss.str();

      auto numPartsBefore = regionPartitions[regionId].size();

      auto rePartitionRet = rePartition(
      regionId, regionGraphs[regionId], 
      regionWorkDirectory[regionId], 
      regionPartitions[regionId]);
      if (failed(rePartitionRet)) return failure();

      auto numPartsAfter = regionPartitions[regionId].size();
      assert(numPartsAfter >= numPartsBefore);
      if (numPartsAfter == numPartsBefore) converged = true;
      
      rePartIterationCount++;
    }

    msgOss.str("");
    msgOss.clear();
    msgOss << "Region " << regionId << " repartition took " << rePartIterationCount << " iterations, num partitions increased from " 
        << rawPartitionCount << " to " << regionPartitions[regionId].size() << "\n";
    llvm::outs() << msgOss.str();

    if (!converged) {
      llvm::errs() << "Fail to limit partition size by repartition!\n";
      return failure();
    }

    return ret;
  });

  bgDumpThread.join();

  if (failed(ret)) return ret;

  llvm::outs() << "====================Partitioning Statistics====================\n";
  for (size_t regionId = 0; regionId < numRegions; regionId++) {
    auto stat = getPartitionStatistics(regionId);

    llvm::outs() << "Statistics for region " << regionId << ":\n";
    printPartitionStatistics(stat);
  }

  auto levelize_stat = levelizeAllPartitions(context);
  assert(succeeded(levelize_stat));

  llvm::outs() << "====================Schedule And Codegen====================\n";

  auto schedule_start = std::chrono::high_resolution_clock::now();
  schedule(graph);
  auto schedule_end = std::chrono::high_resolution_clock::now();
  auto schedule_duration = std::chrono::duration_cast<std::chrono::milliseconds>(schedule_end - schedule_start).count();
  llvm::outs() << "Scheduling took " << schedule_duration << "ms\n";

  // for (size_t partId = 0; partId < codeGenInfo.partitionInfo.size(); partId++) {
  //   const auto &eachPart = codeGenInfo.partitionInfo[partId];
  //   llvm::outs() << "Partition " << partId << "\n";
  //   eachPart.opStatistics.print();
  // }

  SchedulerBase::fillDebugInfo();

  return success();
}

void RepCutPartitioner::dumpGraphToFile(const PartitioningGraph &g, std::string fileName) const {
  auto ofs = std::ofstream(fileName);

  ofs << boost::num_edges(g) << ' ' << boost::num_vertices(g) << "\n";

  auto numVtxes = boost::num_vertices(g);
  for (uint32_t vtx = 0; vtx < numVtxes; vtx++) {
    ofs << stringifyCGToucanOPName(g[vtx].toucanOpName) << ' ' << g[vtx].weight;
    
    for (auto ei = boost::out_edges(vtx, g); ei.first != ei.second; ++ei.first) {
      auto u = boost::target(*ei.first, g);
      ofs << ' ' << u;
    }
    ofs << "\n";
  }
}

LogicalResult RepCutPartitioner::callRepCutAndWait(uint32_t nParts, float target_ib, const std::string &graphFile, const std::filesystem::path &workingDirectory) {
  // auto programLocation = boost::dll::program_location().string();
  const char* rcpBinary = "rcp";
  auto ibString = std::to_string(target_ib);
  auto nPartsString = std::to_string(nParts);
  std::string repcutPrintLogPath = workingDirectory / repcutConsoleLogFileName;

  llvm::StringRef args[] = {rcpBinary, "--target_ib", ibString, "--nparts", nPartsString, "--graph_file", graphFile, "--work_directory", workingDirectory.c_str(), "--log_level", "debug"};

  std::optional<llvm::StringRef> redirects[] = {
    std::nullopt,
    repcutPrintLogPath,
    repcutPrintLogPath
  };

  auto rcpExe = llvm::sys::findProgramByName(rcpBinary);
  if (!rcpExe) {
    llvm::errs() << "Cannot find rcp. Please ensure it's in your PATH!\n";
    return failure();
  }


  int result = llvm::sys::ExecuteAndWait(*rcpExe, args, std::nullopt, redirects);

  if (result != 0) {
    llvm::errs() << "RepCut partitioner returns non-zero code: " << result << "\n";
    llvm::errs() << rcpExe.get() << " ";
    for (const auto &eachArg: args) {
      llvm::errs() << eachArg << " ";
    }
    llvm::errs() << "\n";

    return failure();
  }

  return success();
}

static mlir::SmallVector<uint32_t> parseLine(const std::string &line) {
  mlir::SmallVector<uint32_t> ret;
  std::vector<std::string> tokens;
  boost::split(tokens, line, boost::is_any_of(std::string(1, ',')));
  for (auto &token: tokens) {
    if (!token.empty()) {
      ret.push_back(std::stoi(token));
    }
  }

  return ret;
}


mlir::LogicalResult RepCutPartitioner::parseRepCutResult(uint32_t nParts, const std::string &resultFile, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partitions) {
  auto ifs = std::ifstream(resultFile);
  partitions.clear();

  if (!ifs.is_open()) {
    llvm::errs() << "Fail to open repcut output file " << resultFile << "\n";
    return failure();
  }

  std::string line;
  while (std::getline(ifs, line)) {
    partitions.push_back(parseLine(line));
  }

  return success();
}

static float calculate_ib_factor(mlir::SmallVector<uint32_t>& dat) {
  uint32_t total = std::accumulate(dat.begin(), dat.end(), static_cast<uint32_t>(0));
  uint32_t max = *std::max_element(dat.begin(), dat.end());
  uint32_t avg = total / dat.size();

  return static_cast<float>(max - avg) / static_cast<float>(avg);
}

RepCutPartitioningStatistics RepCutPartitioner::getPartitionStatistics(uint32_t regionId) {
  auto parts = regionPartitions[regionId];
  auto &g = regionGraphs[regionId];
  auto graphSize = boost::num_vertices(regionGraphs[regionId]);
  uint32_t graphWeight = 0;

  for (auto vi = boost::vertices(g); vi.first != vi.second; ++vi.first) {
    graphWeight += g[*vi.first].weight;
  }

  RepCutPartitioningStatistics stats;
  for (auto &ep: parts) {
    stats.partSize.push_back(ep.size());
    auto partWeight = getPartWeight(ep, g);
    stats.partWeight.push_back(partWeight);
  }
  stats.graphSize = graphSize;
  stats.graphWeight = graphWeight;

  stats.sizeIBFactor = calculate_ib_factor(stats.partSize);
  stats.weightIBFactor = calculate_ib_factor(stats.partWeight);
  stats.sizeReplicationRate = static_cast<float>(std::accumulate(stats.partSize.begin(), stats.partSize.end(), static_cast<uint32_t>(0)) - graphSize) / static_cast<float>(graphSize);
  stats.weightReplicationRate = static_cast<float>(std::accumulate(stats.partWeight.begin(), stats.partWeight.end(), static_cast<uint32_t>(0)) - graphWeight) / static_cast<float>(graphWeight);

  return stats;
}

void RepCutPartitioner::printPartitionStatistics(const RepCutPartitioningStatistics &stats) {
  llvm::outs() << "Partition size:";
  for (auto &es: stats.partSize) {
    llvm::outs() << " " << es;
  }
  llvm::outs() << "\nPartition weight:";
  for (auto &es: stats.partWeight) {
    llvm::outs() << " " << es;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "\nSize replication rate " 
      << stats.sizeReplicationRate
      << ", ib factor "
      << stats.sizeIBFactor
      << "\nWeight replication rate "
      << stats.weightReplicationRate
      << ", ib factor "
      << stats.weightIBFactor
      << "\n"; 
  llvm::outs() << oss.str();
}


LogicalResult RepCutPartitioner::workerFunc(const PartitioningGraph &graph, std::filesystem::path workDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, uint32_t nParts) {
  if (!std::filesystem::exists(workDirectory)) {
    if (!std::filesystem::create_directories(workDirectory)) {
      llvm::errs() << "Fail to create directory " << workDirectory << "\n";
      return failure();
    }
  }

  /* Should not have cycle. This code is for debugging purpose */
  // bool hasCycle = partitioningGraphHasCycle(graph);
  // if (hasCycle) {
  //   llvm::errs() << "Graph has cycle!\n";
  //   return failure();
  // }

  // Save graph to file
  std::filesystem::path graphPath = workDirectory / graphFileName;
  std::filesystem::path repcutOutputPath = workDirectory / repcutOutputFileName;

  dumpGraphToFile(graph, graphPath);

  auto partitionSucc = callRepCutAndWait(nParts, targetIb, graphPath, workDirectory);
  if (failed(partitionSucc)) {
    llvm::errs() << "RepCut partitioner returns error\n";
    return failure();
  }

  auto parseSucc = parseRepCutResult(nParts, repcutOutputPath, partOutput);
  if (failed(parseSucc)) {
    llvm::errs() << "Fail to parse RepCut result\n";
    return failure();
  }

  return success();
}

static PartitioningGraph createNewGraphFromPartition(const PartitioningGraph &graph, const mlir::SmallVector<uint32_t> &part, mlir::SmallVector<uint32_t> &graphNewIdToOldId) {
  PartitioningGraph newGraph;
  mlir::DenseMap<uint32_t, uint32_t> graphOldIdToNewId;

  graphNewIdToOldId.clear();
  graphNewIdToOldId.resize(part.size(), UINT32_MAX);
  graphOldIdToNewId.reserve(part.size());

  // copy vertices
  for (const auto eachOldVtx: part) {
    const auto &vp = graph[eachOldVtx];
    auto newVtx = boost::add_vertex(vp, newGraph);

    assert(newVtx < part.size());
    graphNewIdToOldId[newVtx] = eachOldVtx;
    graphOldIdToNewId[eachOldVtx] = newVtx;
  }

  assert(boost::num_vertices(newGraph) == graphNewIdToOldId.size());

  // copy edges
  for (const auto eachOldVtx: part) {
    auto out_edge_range = boost::out_edges(eachOldVtx, graph);
    for (auto ei = out_edge_range.first; ei != out_edge_range.second; ei++) {
      auto edgeSource = eachOldVtx;
      auto edgeTarget = boost::target(*ei, graph);

      assert(edgeSource != edgeTarget && "Should have no loop");

      if (graphOldIdToNewId.contains(edgeTarget)) {
        auto edgeSourceNewId = graphOldIdToNewId[edgeSource];
        auto edgeTargetNewId = graphOldIdToNewId[edgeTarget];

        boost::add_edge(edgeSourceNewId, edgeTargetNewId, newGraph);
      }
    }
  }

  return newGraph;
}

mlir::LogicalResult RepCutPartitioner::rePartition(uint32_t regionId, const PartitioningGraph &graph, std::filesystem::path regionWorkDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput) {

  mlir::SmallVector<mlir::SmallVector<uint32_t>> partitions;
  mlir::SmallVector<uint32_t> partsNeedRepartition;

  uint32_t oldPartId = 0;
  for (const auto &eachPart: partOutput) {
    auto partWeight = getPartWeight(eachPart, graph);
    if (partWeight <= PARTITION_MAX_WEIGHT) {
      // a leagel sized partition
      partitions.push_back(eachPart);
    } else {
      partsNeedRepartition.push_back(oldPartId);
      // Need re-partition
    }
    oldPartId++;
  }

  for (auto oldPartId: partsNeedRepartition) {
    // TODO: This can be parallel
    const auto &eachPart = partOutput[oldPartId];
    auto partWeight = getPartWeight(eachPart, graph);

    mlir::SmallVector<uint32_t> graphNewIdToOldId;
    auto newGraph = createNewGraphFromPartition(graph, eachPart, graphNewIdToOldId);
    assert(!graphNewIdToOldId.empty());
    mlir::SmallVector<mlir::SmallVector<uint32_t>> newPartitions;

    std::ostringstream oss;
    oss << "rePartition_" << oldPartId;
    std::string dirName = oss.str();
    auto workDir = regionWorkDirectory / dirName;

    int targetNumPartitions = (partWeight / REPARTITION_PREFERRED_WEIGHT) + 1;

    std::ostringstream msgOss;
    msgOss << "Region " << regionId << " part " << oldPartId
        << ": original size " << partWeight 
        << ", target num parts: " << targetNumPartitions << "\n";
    
    auto ret = workerFunc(newGraph, workDir, newPartitions, targetNumPartitions);
    if (failed(ret)) {
      msgOss << "Error on re-partition!\n";
      llvm::errs() << msgOss.str();
      return failure();
    }

    msgOss << "  Result: ";

    for (const auto &eachNewPartition: newPartitions) {
      auto newPartWeight = getPartWeight(eachNewPartition, newGraph);
      msgOss << newPartWeight << " ";

      mlir::SmallVector<uint32_t> partInOldVtxes;
      partInOldVtxes.reserve(eachNewPartition.size());
      for (const auto &eachNewVtx: eachNewPartition) {
        assert(eachNewVtx < graphNewIdToOldId.size());
        auto oldVtx = graphNewIdToOldId[eachNewVtx];
        partInOldVtxes.push_back(oldVtx);
      }
      partitions.push_back(std::move(partInOldVtxes));
    }
    msgOss << "\n";
    llvm::outs() << msgOss.str();
  }

  std::swap(partitions, partOutput);
  return success();
}