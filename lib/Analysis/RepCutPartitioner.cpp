

#include "mlir/IR/Threading.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanConfigs.h"

#include "toucan/RepCutPartitioner.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Program.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <chrono>

#include <thread>
#include <sstream>
#include <iomanip>

#include <numeric>
#include <vector>
#include <filesystem>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

static bool isDirectoryExists(const std::string &dir) {
  return std::filesystem::exists(dir);
}

static bool createDirectoryIfNotExists(const std::string dir) {
  if (!isDirectoryExists(dir)) {
    bool created = std::filesystem::create_directory(dir);
    return created;
  }
  return true;
}

static uint32_t getPartWeight(const mlir::SmallVector<uint32_t> &part, const PartitioningGraph &graph) {
  uint32_t weight = 0;
  for (const auto eachVtx: part) {
    weight += graph[eachVtx].weight;
  }
  return weight;
};

static uint64_t getGraphTotalWeight(const PartitioningGraph &graph) {
  uint64_t totalWeight = 0;
  auto vtxRange = boost::vertices(graph);
  for (auto it = vtxRange.first; it != vtxRange.second; it++) {
    auto vtxId = *it;
    auto vtxWeight = graph[vtxId].weight;
    totalWeight += vtxWeight;
  }
  return totalWeight;
}


void RepCutPartitioner::setPartitionTarget(float partSizeRatio, int targetGPUSMCount) {

  auto &graph = *_graph;
  auto eachRegionWeight = getGraphTotalWeight(graph);

  if (partSizeRatio < 0.01 || partSizeRatio > 1.0) {
    // Need decide part ratio
    partSizeRatio = REPCUT_PART_SIZE_RATIO_START;

    while (partSizeRatio < REPCUT_PART_SIZE_RATIO_MAX) {

      int part_max_weight = PARTITION_MAX_WEIGHT * partSizeRatio;
      int part_preferred_weight = PARTITION_PREFERRED_WEIGHT * partSizeRatio;
      auto preferredPartCount = (eachRegionWeight / part_preferred_weight) + 1;

      if (preferredPartCount <= (REPCUT_AUTO_PART_CNT_GPU_SM_MARGIN * targetGPUSMCount)) {
        // ok we can take this
        PARTITION_MAX_WEIGHT = part_max_weight;
        PARTITION_PREFERRED_WEIGHT = part_preferred_weight;
        break;
      }

      partSizeRatio += REPCUT_PART_SIZE_RATIO_STEP;
    }

    llvm::outs() << "Choose " << std::to_string(partSizeRatio) << " as part size ratio\n";
  } else {
    PARTITION_MAX_WEIGHT = PARTITION_MAX_WEIGHT * partSizeRatio;
    PARTITION_PREFERRED_WEIGHT = PARTITION_PREFERRED_WEIGHT * partSizeRatio;
    llvm::outs() << "User override: part size ratio is " << partSizeRatio << "\n";
  }
  auto preferredPartCount = (eachRegionWeight / PARTITION_PREFERRED_WEIGHT) + 1;
  llvm::outs() << "Graph total weight is " << eachRegionWeight << ", preferred Part count is " << preferredPartCount << "\n";

  repcutTargetPartCount = preferredPartCount;
  
}

LogicalResult RepCutPartitioner::_partition(mlir::MLIRContext *context) {
  if (!isDirectoryExists(outputDirectory)) {
    llvm::errs() << "Output directory does not exists! (" << outputDirectory << ")\n";
    return failure();
  }

  assert(_graph != nullptr);

  {
    auto start = std::chrono::high_resolution_clock::now();
    auto maxThreads = context->getNumThreads();

    // No need call repcut
    auto targetNumParts = repcutTargetPartCount;
    if (targetNumParts == 1) {
      repcutPartitions.clear();
      repcutPartitions.emplace_back();
      repcutPartitions.back().reserve(boost::num_vertices(*_graph));
      for (auto vtx: boost::make_iterator_range((boost::vertices(*_graph)))) {
        repcutPartitions[0].push_back(vtx);
      }

      return success();
    }

    auto ret = workerFunc(
      *_graph, 
      outputDirectory, 
      repcutPartitions, 
      repcutTargetPartCount,
      maxThreads);
    if (failed(ret)) return ret;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::ostringstream msgOss;
    msgOss << "Graph spend " << duration << "ms\n";
    llvm::outs() << msgOss.str();

    uint32_t rePartIterationCount = 0;
    bool converged = false;
    uint32_t rawPartitionCount = repcutPartitions.size();
    uint32_t numPartsBefore;
    keepRepartitionMayUseless = false;
    while ((rePartIterationCount <= rePartitionMaxIterations) && (!converged)) {
      msgOss.str("");
      msgOss.clear();
      msgOss << "> Repartition iteration " << rePartIterationCount << "\n";
      llvm::outs() << msgOss.str();

      numPartsBefore = repcutPartitions.size();

      auto rePartitionRet = rePartition(
        context, 
        *_graph, 
        outputDirectory, 
        repcutPartitions,
        rePartIterationCount,
        numPartsBefore);
      if (failed(rePartitionRet)) return failure();

      auto numPartsAfter = repcutPartitions.size();
      assert(numPartsAfter >= numPartsBefore);
      if (numPartsAfter == numPartsBefore) {
        converged = true;
      }

      if (keepRepartitionMayUseless) converged = true;
      
      rePartIterationCount++;
    }

    msgOss.str("");
    msgOss.clear();
    msgOss << "Repartition took " << rePartIterationCount << " iterations, num partitions increased from " 
        << rawPartitionCount << " to " << repcutPartitions.size() << "\n";
    llvm::outs() << msgOss.str();

    if (!converged) {
      llvm::errs() << "Fail to limit partition size by repartition!\n";
      return failure();
    }

    // sort
    llvm::outs() << "Sort partitions by weight\n";
    std::sort(repcutPartitions.begin(), repcutPartitions.end(), [&](const auto &a, const auto &b) {
      auto a_weight = getPartWeight(a, *_graph);
      auto b_weight = getPartWeight(b, *_graph);
      return a_weight > b_weight;
    });

  }




  llvm::outs() << "====================Partitioning Statistics====================\n";
  {
    auto stat = getPartitionStatistics();

    llvm::outs() << "Statistics:\n";
    printPartitionStatistics(stat);
  }

  return success();
}


bool are_ids_consecutive(const PartitioningGraph& g) {
  auto [begin, end] = vertices(g);
  for (size_t i = 0; begin != end; ++begin, ++i) {
    if (*begin != i) return false;
  }
  return true;
}


// Dump graph for RepCut use
// Note: Different from dumpSinglePartitionToFile!!!!!
// Merge all duplicated edges (does not allow parallel edge)
LogicalResult RepCutPartitioner::dumpGraphToFile(const PartitioningGraph &g, std::string fileName) {

  auto oss = std::ostringstream();

  auto numVtxes = boost::num_vertices(g);
  // Note: numEdges may not be same with boost::num_edges(g). See later for reason
  uint64_t numEdges = 0;



  assert(are_ids_consecutive(g));
  for (uint32_t vtx = 0; vtx < numVtxes; vtx++) {
    auto tOpName = g[vtx].toucanOpName;


    mlir::SmallVector<uint32_t> outNeighs;
    // dedup same edge
    mlir::DenseSet<uint32_t> outNeighsSet;
    for (auto ei = boost::out_edges(vtx, g); ei.first != ei.second; ++ei.first) {
      auto v = boost::target(*ei.first, g);
      outNeighsSet.insert(v);
    }
    outNeighs.assign(outNeighsSet.begin(), outNeighsSet.end());



    numEdges += outNeighs.size();
    oss << stringifyCGToucanOPName(tOpName);
    oss << ' ' << g[vtx].weight;
    for (const auto en: outNeighs) {
      oss << ' ' << en;
    }
    oss << "\n";
  }


  auto ofs = std::ofstream(fileName, std::ios::out | std::ios::trunc);

  if (!ofs.is_open()) {
    llvm::errs() << "Cannot open file for write: " << fileName << "\n";
    return failure();
  }

  ofs << numEdges << ' ' << numVtxes << "\n";
  ofs << oss.str();

  ofs.close();
  return success();
}


int RepCutPartitioner::decideRepCutNumThreads(int maxThreads, int numTargetPartitions) {
  return std::min(maxThreads, static_cast<int>(numTargetPartitions / 8) + 1);
}

LogicalResult RepCutPartitioner::callRepCutAndWait(uint32_t nParts, float target_ib, const std::string &graphFile, const std::filesystem::path &workingDirectory, int maxThreads) {
  // auto programLocation = boost::dll::program_location().string();
  const char* rcpBinary = "rcp";
  auto ibString = std::to_string(target_ib);
  auto nPartsString = std::to_string(nParts);
  auto numThreads = decideRepCutNumThreads(maxThreads, nParts);
  auto numThreadsString = std::to_string(numThreads);
  std::string repcutPrintLogPath = workingDirectory / repcutConsoleLogFileName;

  if (!std::filesystem::exists(graphFile)) {
    llvm_unreachable("RepCut partitioner input file doesn't exists! This should not happen");
  }

  llvm::StringRef args[] = {
    rcpBinary, 
    "--target_ib", ibString, 
    "--nparts", nPartsString, 
    "--graph_file", graphFile, 
    "--work_directory", workingDirectory.c_str(), 
    "--log_level", "trace", 
    "--threads", numThreadsString
  };

  std::ostringstream oss;
  for (const auto &ea: args) {
    oss << ea.str() << " ";
  }
  llvm::outs() << "RCP args: " << oss.str() << "\n";

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

RepCutPartitioningStatistics RepCutPartitioner::getPartitionStatistics() {
  auto parts = repcutPartitions;
  auto &g = *_graph;
  auto graphSize = boost::num_vertices(*_graph);
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


LogicalResult RepCutPartitioner::workerFunc(const PartitioningGraph &graph, std::filesystem::path workDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, uint32_t nParts, int maxThreads) {
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

  auto ret = dumpGraphToFile(graph, graphPath);
  if (failed(ret)) return ret;

  auto partitionSucc = callRepCutAndWait(nParts, targetIb, graphPath, workDirectory, maxThreads);
  if (failed(partitionSucc)) {
    llvm::errs() << "RepCut partitioner returns error\n";
    return failure();
  }

  auto parseSucc = parseRepCutResult(nParts, repcutOutputPath, partOutput);
  if (failed(parseSucc)) {
    llvm::errs() << "Fail to parse RepCut result\n";
    return failure();
  }

  auto stat = getPartitionStatistics();

  llvm::outs() << "Statistics:\n";
  printPartitionStatistics(stat);
  return success();
}

static PartitioningGraph createNewGraphFromPartition(const PartitioningGraph &graph, const mlir::SmallVector<uint32_t> &part, mlir::SmallVector<uint32_t> &graphNewIdToOldId) {
  assert(part.size() != 0);
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

mlir::LogicalResult RepCutPartitioner::rePartition(mlir::MLIRContext *context, const PartitioningGraph &graph, std::filesystem::path regionWorkDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, const uint32_t iterId, uint32_t &previousRePartitionInputNumParts) {

  mlir::SmallVector<mlir::SmallVector<uint32_t>> partitions;
  mlir::SmallVector<uint32_t> partsNeedRepartition;
  // uint32_t partsNeedRepartitionWeights = 0;

  assert(PARTITION_MAX_WEIGHT > PARTITION_PREFERRED_WEIGHT);
  for (uint32_t oldPartId = 0; oldPartId < partOutput.size(); oldPartId++) {
    const auto &eachPart = partOutput[oldPartId];
    auto partWeight = getPartWeight(eachPart, graph);

    llvm::dbgs() << "Part " << oldPartId << " weight is " << partWeight << ", part max weight " << PARTITION_MAX_WEIGHT << "\n";
    if (partWeight <= PARTITION_MAX_WEIGHT) {
      // a leagel sized partition
      partitions.push_back(eachPart);
    } else {
      // partsNeedRepartitionWeights += partWeight;
      partsNeedRepartition.push_back(oldPartId);
    }
  }
  if (partsNeedRepartition.empty()) return success();

  
  uint32_t numOldParts = partsNeedRepartition.size();

  llvm::outs() << "previous repart input num parts " << previousRePartitionInputNumParts << ", numoldparts " << numOldParts << "\n";
  if (previousRePartitionInputNumParts != 0 && previousRePartitionInputNumParts < numOldParts) {
    llvm::outs() << "Previous repartition fail to reduce imbalanced partition count. Stop repartition!\n";
    keepRepartitionMayUseless = true;
    return success();
  }
  previousRePartitionInputNumParts = numOldParts;
  auto targetNumNewParts = std::max(static_cast<uint32_t>(numOldParts * REPARTITION_SIZE_TARGET_RATIO), numOldParts + 1);
  mlir::SmallVector<uint32_t> allNodesInNeedRepartition;
  
  {
    // Merge all nodes into a single partition, dedup
    mlir::DenseSet<uint32_t> allNodesInNeedRepartitionSet;
    size_t expectedNodeCount = 0;
    for (const auto &partId: partsNeedRepartition) {
      expectedNodeCount += partOutput[partId].size();
    }
    allNodesInNeedRepartitionSet.reserve(expectedNodeCount);
    for (const auto &partId: partsNeedRepartition) {
      const auto &eachPart = partOutput[partId];
      allNodesInNeedRepartitionSet.insert(eachPart.begin(), eachPart.end());
    }
    assert(allNodesInNeedRepartitionSet.size() != 0);

    allNodesInNeedRepartition.reserve(allNodesInNeedRepartitionSet.size());
    for (const auto &ev: allNodesInNeedRepartitionSet) {
      allNodesInNeedRepartition.push_back(ev);
    }

    assert(allNodesInNeedRepartition.size() != 0);
  }
  


  auto partWeight = getPartWeight(allNodesInNeedRepartition, graph);

  mlir::SmallVector<uint32_t> graphNewIdToOldId;
  auto newGraph = createNewGraphFromPartition(graph, allNodesInNeedRepartition, graphNewIdToOldId);
  assert(!graphNewIdToOldId.empty());
  mlir::SmallVector<mlir::SmallVector<uint32_t>> newPartitions;

  std::ostringstream oss;
  oss << "rePartition_" << iterId;
  std::string dirName = oss.str();
  auto workDir = regionWorkDirectory / dirName;

  auto created = createDirectoryIfNotExists(workDir);
  if (!created) return failure();


  std::ostringstream msgOss;
  msgOss << "Repartition iteration " << iterId
      << ": original weight " << partWeight 
      << ", old num parts: " << partsNeedRepartition.size()
      << ", target num parts: " << targetNumNewParts << "\n";
  llvm::outs() << msgOss.str();
  msgOss.str("");

  auto maxThreads = context->getNumThreads();
  
  auto ret = workerFunc(newGraph, workDir, newPartitions, targetNumNewParts, maxThreads);
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

  std::swap(partitions, partOutput);
  return success();
}