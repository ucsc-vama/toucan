
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"
#include "toucan/ToucanConfigs.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
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

static uint64_t getRegionWeight(const PartitioningGraph &graph) {
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
  auto numRegions_expected = cutPoints.size() + 1;
  auto numRegions = regionGraphs.size();
  assert(numRegions > 0);
  if (numRegions != numRegions_expected) {
    llvm::dbgs() << "Expected num reigons " << numRegions_expected << ", real num regions " << numRegions << "\n";
  }

  for (size_t regionId = 0; regionId < numRegions; regionId++) {
    auto &regionGraph = regionGraphs[regionId];
    auto eachRegionWeight = getRegionWeight(regionGraph);


    if (partSizeRatio < 0.1 || partSizeRatio > 1.0) {
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
          llvm::outs() << "Choose " << partSizeRatio << " as part size ratio\n";
          break;
        }

        partSizeRatio += REPCUT_PART_SIZE_RATIO_STEP;
      }
    } else {
      llvm::outs() << "User override: part size ratio is " << partSizeRatio << "\n";
    }

    auto preferredPartCount = (eachRegionWeight / PARTITION_PREFERRED_WEIGHT) + 1;
    llvm::outs() << "Preferred Part count for region " << regionId << " is " << preferredPartCount << "\n";
    regionPartitionNumbers.push_back(preferredPartCount);
  }
}

LogicalResult RepCutPartitioner::_partition(mlir::MLIRContext *context, DesignGraph &graph) {
  if (!isDirectoryExists(outputDirectory)) {
    llvm::errs() << "Output directory does not exists! (" << outputDirectory << ")\n";
    return failure();
  }

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
  originalVectorElementsMapForEachRegion.resize(numRegions);

  mlir::SmallVector<uint32_t> regionIds(numRegions);
  std::iota(regionIds.begin(), regionIds.end(), 0);

  // llvm::outs() << "Start partitioning\n";
  llvm::outs() << "====================Partitioning====================\n";

  // Save graph to file for debug purpose.
  /*
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
  */

  auto ret = mlir::failableParallelForEach(context, regionIds.begin(), regionIds.end(), [&](uint32_t regionId) {
    auto start = std::chrono::high_resolution_clock::now();
    auto maxThreads = context->getNumThreads();

    auto created = createDirectoryIfNotExists(regionWorkDirectory[regionId]);
    if (!created) return failure();

    std::filesystem::path graphVectorDeclInfoPath = regionWorkDirectory[regionId] / graphVectorDeclInfoFileName;
    auto ret = collectAndDumpGraphVectorDeclInfoToFile(regionId, regionGraphs[regionId], graphVectorDeclInfoPath);
    if (failed(ret)) return ret;


    // No need call repcut
    auto targetNumParts = regionPartitionNumbers[regionId];
    if (targetNumParts == 1) {
      auto &partOutput = regionPartitions[regionId];
      partOutput.clear();
      partOutput.emplace_back();
      partOutput.back().reserve(boost::num_vertices(regionGraphs[regionId]));
      for (auto vtx: boost::make_iterator_range((boost::vertices(regionGraphs[regionId])))) {
        partOutput[0].push_back(vtx);
      }

      return success();
    }

    ret = workerFunc(
      regionGraphs[regionId], 
      regionWorkDirectory[regionId], 
      regionPartitions[regionId], 
      regionPartitionNumbers[regionId],
      maxThreads);
    if (failed(ret)) return ret;

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
        context, 
        regionId, regionGraphs[regionId], 
        regionWorkDirectory[regionId], 
        regionPartitions[regionId],
        rePartIterationCount);
      if (failed(rePartitionRet)) return failure();

      auto numPartsAfter = regionPartitions[regionId].size();
      assert(numPartsAfter >= numPartsBefore);
      if (numPartsAfter == numPartsBefore) {
        converged = true;
      } else {
        // num parts changed
        regionPartitionNumbers[regionId] = numPartsAfter;
      }
      
      rePartIterationCount++;
    }

    msgOss.str("");
    msgOss.clear();
    msgOss << "Region " << regionId << " repartition took " << rePartIterationCount << " iterations, num partitions increased from " 
        << rawPartitionCount << " to " << regionPartitions[regionId].size() << "\n";
    llvm::outs() << msgOss.str();

    if (!converged) {
      llvm::errs() << "Region " << regionId << "Fail to limit partition size by repartition!\n";
      return failure();
    }

    return ret;
  });

  // bgDumpThread.join();

  if (failed(ret)) return ret;

  llvm::outs() << "====================Partitioning Statistics====================\n";
  for (size_t regionId = 0; regionId < numRegions; regionId++) {
    auto stat = getPartitionStatistics(regionId);

    llvm::outs() << "Statistics for region " << regionId << ":\n";
    printPartitionStatistics(stat);
  }

  return success();
}

// LogicalResult RepCutPartitioner::_schedule(mlir::MLIRContext *context, DesignGraph &graph) {

//   llvm::outs() << "====================Schedule And Codegen====================\n";

//   auto schedule_start = std::chrono::high_resolution_clock::now();
//   schedule(graph);
//   auto schedule_end = std::chrono::high_resolution_clock::now();
//   auto schedule_duration = std::chrono::duration_cast<std::chrono::milliseconds>(schedule_end - schedule_start).count();
//   llvm::outs() << "Scheduling took " << schedule_duration << "ms\n";

//   // for (size_t partId = 0; partId < codeGenInfo.partitionInfo.size(); partId++) {
//   //   const auto &eachPart = codeGenInfo.partitionInfo[partId];
//   //   llvm::outs() << "Partition " << partId << "\n";
//   //   eachPart.opStatistics.print();
//   // }

//   // Do not fill signal names. Values may now be reclaimed
//   // SchedulerBase::fillDebugInfo(false);

//   return success();
// }

// LogicalResult RepCutPartitioner::partitionAndSchedule(mlir::MLIRContext *context, DesignGraph &graph) {
//   auto ret = _partition(context, graph);
//   if (failed(ret)) return failure();

//   // Levelize
//   auto levelize_stat = levelizeAllPartitions(context);
//   assert(succeeded(levelize_stat));

//   return _schedule(context, graph);
// }

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
LogicalResult RepCutPartitioner::dumpGraphToFile(const PartitioningGraph &g, std::string fileName) const {

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

// Dump graph for MicroPartitioner use
// Note: Different from dumpGraphToFile!!!
// Partially allow parallel edge: parallel edge from VecRead/VecArith implies
//    multiple value read from a merged node
LogicalResult RepCutPartitioner::dumpSinglePartitionToFile(const PartitioningGraph &g, mlir::SmallVector<uint32_t> partNodes, std::string fileName) const {
  mlir::DenseSet<uint32_t> partNodeSet;
  for (auto n: partNodes) {
    partNodeSet.insert(n);
  }

  
  std::ostringstream oss;
  size_t numValidVtxes = 0, numValidEdges = 0;


  for (auto vtx: boost::make_iterator_range((boost::vertices(g)))) {
    if (partNodeSet.contains(vtx)) {
      // valid node
      numValidVtxes++;
      auto tOpName = g[vtx].toucanOpName;

      mlir::SmallVector<uint32_t> outNeighs;

      if (tOpName == CGToucanOPName::VecArith || tOpName == CGToucanOPName::VecRead) {
        // Use parallel edge to represent multiple reads from same merged node (but different ops)
        auto rawOp = g[vtx].op;
        assert(rawOp != nullptr);
        mlir::DenseSet<mlir::Value> allResultValueOfThisNode;

        // Collect output values for the same merged node
        if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(rawOp)) {
          assert(tOpName == CGToucanOPName::VecRead);
          auto vecVal = vecReadOp.getHandle();

          for (auto userOp: vecVal.getUsers()) {
            if (auto vecReadOp = dyn_cast<toucan::VectorReadOp>(userOp)) {
              allResultValueOfThisNode.insert(vecReadOp.getResult());
            }
          }
        } else {
          assert(tOpName == CGToucanOPName::VecArith);
          auto vecArithOp = cast<toucan::VectorArithOp>(rawOp);
          auto vecVal = vecArithOp.getResult();
          for (auto userOp: vecVal.getUsers()) {
            if (auto staticReadOp = dyn_cast<toucan::StaticVectorSegmentReadOp>(userOp)) {
              allResultValueOfThisNode.insert(staticReadOp.getResult());
            }
          }
        }

        mlir::DenseSet<uint32_t> visitedEdges;
        for (auto ei = boost::out_edges(vtx, g); ei.first != ei.second; ++ei.first) {
          auto v = boost::target(*ei.first, g);

          if (visitedEdges.contains(v)) {
            // Dont visit parallel edge, they are recalculated later
            continue;
          }
          visitedEdges.insert(v);

          // Use parallel edge to represent multiple reads from a same merged node
          // This allows MicroPartitioner correctly track read count
          uint32_t readCount = 0;

          mlir::SmallVector<mlir::Operation*> targetRawOps;
          auto _targetRawOp = g[v].op;
          switch (g[v].toucanOpName) {
            case toucan::CGToucanOPName::VecRead: {
              uint32_t thisNodeOpCount = 0;
              auto vecReadOp = cast<toucan::VectorReadOp>(_targetRawOp);
              auto vecVal = vecReadOp.getHandle();

              for (auto userOp: vecVal.getUsers()) {
                if (isa<toucan::VectorReadOp>(userOp)) {
                  targetRawOps.push_back(userOp);
                  thisNodeOpCount++;
                }
              }

              assert(thisNodeOpCount == g[v].opCount);
              break;
            }

            case toucan::CGToucanOPName::MemWrite: {
              uint32_t thisNodeOpCount = 0;
              auto mwOp = cast<toucan::MemWriteOp>(_targetRawOp);
              auto memVal = mwOp.getMem();

              for (auto userOp: memVal.getUsers()) {
                if (isa<toucan::MemWriteOp>(userOp)) {
                  targetRawOps.push_back(userOp);
                  thisNodeOpCount++;
                }
              }

              assert(thisNodeOpCount == g[v].opCount);
              break;
            }
            default: {
              // Note: do nothing for VecArith following VecStaticSegRead, as they only read from producing result of VecArith
              targetRawOps.push_back(_targetRawOp);
            }
          }

          for (auto targetRawOp: targetRawOps) {
            if (targetRawOp == nullptr) {
              readCount = 1;
            } else {
              for (const auto eachOperand: targetRawOp->getOperands()) {
                if (allResultValueOfThisNode.contains(eachOperand)) {
                  readCount += 1;
                }
              }
            }
          }

          assert(readCount >= 1);
          for (size_t i = 0; i < readCount; i++) {
            outNeighs.push_back(v);
          }
        }
      } else {
        // Otherwise, dedup same edge
        mlir::DenseSet<uint32_t> outNeighsSet;
        for (auto ei = boost::out_edges(vtx, g); ei.first != ei.second; ++ei.first) {
          auto v = boost::target(*ei.first, g);
          outNeighsSet.insert(v);
        }
        outNeighs.assign(outNeighsSet.begin(), outNeighsSet.end());
      }


      mlir::SmallVector<uint32_t> validOutNeighs;
      for (auto &target: outNeighs) {
        if (partNodeSet.contains(target)) {
          validOutNeighs.push_back(target);
        }
      }

      numValidEdges += validOutNeighs.size();

      oss << vtx << ' ';
      oss << stringifyCGToucanOPName(tOpName);
      oss << ' ' << g[vtx].weight;
      for (const auto v: validOutNeighs) {
        oss << ' ' << v;
      }
      oss << "\n";
    }
  }

  assert(numValidVtxes == partNodeSet.size());


  auto ofs = std::ofstream(fileName, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    llvm::errs() << "Cannot open file for write: " << fileName << "\n";
    return failure();
  }
  ofs << numValidEdges << ' ' << numValidVtxes << "\n";
  ofs << oss.str();

  ofs.close();

  return success();
}

LogicalResult RepCutPartitioner::dumpAllPartitionsToFile() {
  auto numRegions = regionGraphs.size();
  assert(numRegions >= 1);

  for (size_t regionId = 0; regionId < numRegions; regionId++) {
    auto &parts = regionPartitions[regionId];
    auto numParts = parts.size();
    assert(numParts == regionPartitionNumbers[regionId]);

    auto thisRegionWorkDirectory = regionWorkDirectory[regionId];

    for (size_t partId = 0; partId < parts.size(); partId++) {
      // llvm::outs() << "Dumping region " << regionId << " part " << partId << "\n";

      std::ostringstream oss;
      oss << "dump_part_" << partId << ".graph";
      auto graphFileName = thisRegionWorkDirectory / oss.str();
      
      auto ret = dumpSinglePartitionToFile(regionGraphs[regionId], parts[partId], graphFileName);

      if (failed(ret)) return ret;
    }
  }

  return success();
}

LogicalResult RepCutPartitioner::collectAndDumpGraphVectorDeclInfoToFile(const uint32_t regionId, const PartitioningGraph &g, std::string fileName) {
  assert(originalVectorElementsMapForEachRegion.size() > regionId);
  auto &originalVectorElementsMap = originalVectorElementsMapForEachRegion[regionId];

  originalVectorElementsMap.clear();

  auto ofs = std::ofstream(fileName, std::ios::out | std::ios::trunc);

  if (!ofs.is_open()) {
    llvm::errs() << "Cannot open file for write: " << fileName << "\n";
    return failure();
  }

  ofs << "Format:\nVecDecl_node_id Vector_element_id_0 Vector_element_id_1 ...\n";


  auto numVtxes = boost::num_vertices(g);
  for (uint32_t vtx = 0; vtx < numVtxes; vtx++) {
    auto vtxOpName = g[vtx].toucanOpName;

    if (vtxOpName == CGToucanOPName::VecDecl) {
      // A vector decl
      auto vecDeclOp = dyn_cast<toucan::DefVectorOp>(g[vtx].op);
      assert(vecDeclOp != nullptr);

      mlir::DenseMap<mlir::Value, uint32_t> vecInputValToOpId;
      // check input edges (that ultimately forms the vector)
      for (auto ei = boost::in_edges(vtx, g); ei.first != ei.second; ++ei.first) {
        auto inputNodeId = boost::source(*ei.first, g);
        auto inputNodeOp = g[inputNodeId].op;
        auto inputNodeOpName = g[inputNodeId].toucanOpName;

        assert(inputNodeOpName != CGToucanOPName::VecDecl);

        if (inputNodeOpName == CGToucanOPName::VecRead) {
          // May be a bunch of VecRead
          auto inputVector = cast<toucan::VectorReadOp>(inputNodeOp).getHandle();
          for (auto userOp: inputVector.getUsers()) {
            auto mergedVecReadOp = cast<toucan::VectorReadOp>(userOp);
            auto resultVal = mergedVecReadOp.getResult();

            // It's OK to see same value many times, as we allow parallel edge
            if (vecInputValToOpId.contains(resultVal)) {
              assert(vecInputValToOpId[resultVal] == inputNodeId);
            }

            vecInputValToOpId[resultVal] = inputNodeId;
          }
        } else if (inputNodeOpName == CGToucanOPName::VecArith) {
          for (auto userOp: inputNodeOp->getUsers()) {
            // Each user should be StaticVectorSegmentReadOp
            auto segReadOp = cast<toucan::StaticVectorSegmentReadOp>(userOp);
            auto resultVal = segReadOp.getResult();

            // It's OK to see same value many times, as we allow parallel edge
            if (vecInputValToOpId.contains(resultVal)) {
              assert(vecInputValToOpId[resultVal] == inputNodeId);
            }

            vecInputValToOpId[resultVal] = inputNodeId;
          }
        } else {
          // any better way?
          auto resultVal = inputNodeOp->getUses().begin()->get();

          if (vecInputValToOpId.contains(resultVal)) {
            // Tolerate identical input edges
            assert(vecInputValToOpId[resultVal] == inputNodeId);
          }

          vecInputValToOpId[resultVal] = inputNodeId;
        }
      }

      ofs << vtx;
      assert(!originalVectorElementsMap.contains(vtx));
      originalVectorElementsMap[vtx] = {};
      for (auto inputVal: vecDeclOp.getInputs()) {
        if (!vecInputValToOpId.contains(inputVal)) {
          inputVal.getDefiningOp()->print(llvm::dbgs());
        }
        assert(vecInputValToOpId.contains(inputVal));
        auto sourceVtx = vecInputValToOpId[inputVal];
        ofs << " " << sourceVtx;
        originalVectorElementsMap[vtx].push_back(sourceVtx);
      }
      ofs << "\n";
    }
  }

  ofs.close();
  return success();
}

int RepCutPartitioner::decideRepCutNumThreads(int maxThreads, int numTargetPartitions) {
  return std::min(maxThreads, static_cast<int>(numTargetPartitions / 4) + 1);
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

mlir::LogicalResult RepCutPartitioner::rePartition(mlir::MLIRContext *context, uint32_t regionId, const PartitioningGraph &graph, std::filesystem::path regionWorkDirectory, mlir::SmallVector<mlir::SmallVector<uint32_t>> &partOutput, const uint32_t iterId) {

  mlir::SmallVector<mlir::SmallVector<uint32_t>> partitions;
  mlir::SmallVector<uint32_t> partsNeedRepartition;


  for (uint32_t oldPartId = 0; oldPartId < partOutput.size(); oldPartId++) {
    const auto &eachPart = partOutput[oldPartId];
    auto partWeight = getPartWeight(eachPart, graph);
    if (partWeight <= PARTITION_MAX_WEIGHT) {
      // a leagel sized partition
      partitions.push_back(eachPart);
    } else {
      partsNeedRepartition.push_back(oldPartId);
    }
  }

  if (partsNeedRepartition.empty()) return success();

  
  uint32_t numOldParts = partsNeedRepartition.size();
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
  msgOss << "Region " << regionId << " repartition iteration " << iterId
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