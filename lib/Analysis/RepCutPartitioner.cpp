
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/IR/Threading.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Program.h"
#include <cstdint>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <chrono>

#include <thread>
#include <future>
#include <format>

#include <numeric>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


LogicalResult RepCutPartitioner::partitionAndSchedule(mlir::MLIRContext *context, DesignGraph &graph) {

  // Levelize
  levelizeGraph(graph);

  // Cut into 2 subgraph
  findCutPoints(graph);
  cutGraph(graph);


  auto numVtxes = boost::num_vertices(graph.g);
  auto numRegions = regionGraphs.size();

  for (size_t rid = 0; rid < numRegions; rid++) {
    std::string dirName = std::format("region{}", rid);
    regionWorkDirectory.push_back(outputDirectory / dirName);
  }

  // TODO: for now, simply use fixed number of partitions
  regionPartitionNumbers.resize(numRegions, 4);

  regionPartitions.clear();
  regionPartitions.resize(numRegions);

  mlir::SmallVector<uint32_t> regionIds(numRegions);
  std::iota(regionIds.begin(), regionIds.end(), 0);

  // llvm::outs() << "Start partitioning\n";

  auto ret = mlir::failableParallelForEach(context, regionIds.begin(), regionIds.end(), [&](uint32_t regionId) {
    auto start = std::chrono::high_resolution_clock::now();

    auto ret = workerFunc(
      regionGraphs[regionId], 
      regionWorkDirectory[regionId], 
      regionPartitions[regionId], 
      regionPartitionNumbers[regionId]);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    auto msg = std::format("Graph region {} spend {}ms\n", regionId, duration);
    llvm::outs() << msg;

    return ret;
  });

  // Save graph to file for debug purpose. Can be removed later.
  // dumpGraphToFile(graph.g, wholeGraphPath);

  if (failed(ret)) return ret;

  for (size_t regionId = 0; regionId < numRegions; regionId++) {
    auto stat = getPartitionStatistics(regionId);

    llvm::outs() << "Statistics for region " << regionId << ":\n";
    printPartitionStatistics(stat);
  }

  levelizeAllPartitions(context);

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

  assert(result == 0);

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
    uint32_t partWeight = 0;
    for (auto &ev: ep) {
      partWeight += g[ev].weight;
    }
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
  auto formatStr = std::format("\nSize replication rate {:.3f}, ib factor {:.3f}\nWeight replication rate {:.3f}, ib factor {:.3f}\n", stats.sizeReplicationRate, stats.sizeIBFactor, stats.weightReplicationRate, stats.weightIBFactor);
  llvm::outs() << formatStr;
}


LogicalResult RepCutPartitioner::workerFunc(const PartitioningGraph &graph, std::filesystem::path workDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, uint32_t nParts) {
  if (!std::filesystem::exists(workDirectory)) {
    if (!std::filesystem::create_directories(workDirectory)) {
      llvm::errs() << "Fail to create directory " << workDirectory << "\n";
      return failure();
    }
  }

  // Save graph to file
  std::filesystem::path graphPath = workDirectory / graphFileName;
  std::filesystem::path repcutOutputPath = workDirectory / repcutOutputFileName;

  dumpGraphToFile(graph, graphPath);

  auto partitionSucc = callRepCutAndWait(nParts, targetIb, graphPath, workDirectory);
  if (failed(partitionSucc)) return failure();

  auto parseSucc = parseRepCutResult(nParts, repcutOutputPath, partOutput);
  if (failed(parseSucc)) return failure();

  return success();
}