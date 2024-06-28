#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <optional>
#include <tuple>
#include <vector>
#include <numeric>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;




void MultiRegionScheduler::levelizeGraph(DesignGraph &graph) {
  levelizeWorker(graph.g, graphLevels);

  // debug print

  llvm::outs() << "Graph has " << graphLevels.size() << " levels\n";
  for (size_t levelId = 0; levelId < graphLevels.size(); levelId++) {
    auto &currentLevel = graphLevels[levelId];
    llvm::outs() << "  Level " << levelId << " has " << currentLevel.size() << " verticies\n";
  }
}

void MultiRegionScheduler::findCutPoints(DesignGraph &graph) {
  assert(graphLevels.size() > 1);
  // TODO: Is this the best policy?
  // cut to 3 regions: 0.4, 0.3, rest
  mlir::SmallVector<float> cutRatios = {0.4, 0.3};

  uint32_t graphSize = boost::num_vertices(graph.g);
  uint32_t regionTarget = graphSize * cutRatios[0];

  uint32_t cutLevel = 0;
  uint32_t currentRegionSize = 0;
  uint32_t cursor = 0;

  cutPoints.clear();

  for (auto &eachLevel: graphLevels) {
    currentRegionSize += eachLevel.size();
    if (currentRegionSize >= regionTarget) {
      llvm::outs() << "Cut after level " << cutLevel << "\n";

      cutPoints.push_back(cutLevel);
      currentRegionSize = 0;
      cursor++;
      regionTarget = (cutRatios.size() > cursor) ? (cutRatios[cursor] * graphSize) : (graphSize + 1);
    }
    cutLevel++;
  }

  return;
}

void MultiRegionScheduler::cutGraph(DesignGraph &graph) {
  assert(graphLevels.size() > 1);
  assert(!cutPoints.empty());
  int32_t prevCutPoint = -1;
  for (auto ep: cutPoints) {
    assert(ep < graphLevels.size());
    assert(static_cast<int32_t>(ep) > prevCutPoint);
    prevCutPoint = static_cast<int32_t>(ep);
  }

  uint32_t graphSize = boost::num_vertices(graph.g);

  // Map between old vertex to vertex in every region.
  // A node can be in only 1 region, thus mixing them together is fine.
  mlir::SmallVector<uint32_t> vtxIdToNewId;
  mlir::SmallVector<uint32_t> vtxIdToRegionId;
  mlir::SmallVector<mlir::SmallVector<uint32_t>> regionVtxes;

  vtxIdToNewId.assign(graphSize, UINT32_MAX);
  vtxIdToRegionId.assign(graphSize, UINT32_MAX);
  regionVtxes.emplace_back();


  size_t currentRegionId = 0;
  auto nextCutPoint = cutPoints[0];
  for (size_t levelId = 0; levelId < graphLevels.size(); levelId++) {
    const auto currentLevel = graphLevels[levelId];

    if (levelId > nextCutPoint) {
      currentRegionId++;
      regionVtxes.emplace_back();
      nextCutPoint = (currentRegionId >= cutPoints.size()) ? graphLevels.size() : cutPoints[currentRegionId];
    }
    
    regionVtxes.back().insert(regionVtxes.back().end(), currentLevel.begin(), currentLevel.end());
    assert(currentRegionId + 1 == regionVtxes.size());
    // llvm::dbgs() << "Put level " << levelId << " to region " << currentRegionId << "\n";
  }
  

  // copy nodes for every new graph
  uint32_t regionId = 0;
  for (const auto &eachRegionVtxes: regionVtxes) {
    regionGraphs.emplace_back();
    auto &rg = regionGraphs.back();
    for (const auto &ev: eachRegionVtxes) {
      auto newVertex = boost::add_vertex(graph.g[ev], rg);

      assert(vtxIdToNewId[ev] == UINT32_MAX);
      assert(vtxIdToRegionId[ev] == UINT32_MAX);
      vtxIdToNewId[ev] = newVertex;
      vtxIdToRegionId[ev] = regionId;
    }

    llvm::outs() << "Before add extra IO, region " << regionId << " has " << boost::num_vertices(rg) << " verticies\n";

    regionId++;
  }



  // Add edges
  auto numRegions = regionGraphs.size();

  mlir::DenseMap<uint32_t, uint32_t> writerToExchangeValId;
  mlir::SmallVector<mlir::DenseMap<uint32_t, uint32_t>> exchangeValIdToReader(numRegions);
  uint32_t numExchangeWrite = 0;
  uint32_t numExchangeRead = 0;
  // mlir::SmallVector<uint32_t> numExgWriteInRegion(numRegions);
  // mlir::SmallVector<uint32_t> numExgReadInRegion(numRegions);

  mlir::SmallVector<mlir::SmallVector<uint32_t>> exgWriteInRegion(numRegions);
  mlir::SmallVector<mlir::SmallVector<uint32_t>> exgReadInRegion(numRegions);

  for (size_t i = 0; i < numRegions; i++) {
    exgWriteInRegion.emplace_back();
    exgReadInRegion.emplace_back();
  }


  auto rawEdges = boost::edges(graph.g);
  for (auto ei = rawEdges.first; ei != rawEdges.second; ++ei) {
    auto edgeSource = boost::source(*ei, graph.g);
    auto edgeTarget = boost::target(*ei, graph.g);
    auto srcNewId = vtxIdToNewId[edgeSource];
    auto dstNewId = vtxIdToNewId[edgeTarget];
    auto srcRegion = vtxIdToRegionId[edgeSource];
    auto dstRegion = vtxIdToRegionId[edgeTarget];

    if (srcRegion == dstRegion) {
      // Internal edge
      boost::add_edge(srcNewId, dstNewId, regionGraphs[srcRegion]);
    } else {
      // An edge that cross (possibly multipe) regions
      assert(srcRegion < dstRegion);

      if (!writerToExchangeValId.contains(edgeSource)) {
        // Allocate a new exchange val, if it's not already exist
        CGExchangeValueMetaInfo valInfo;
        valInfo.writerId = edgeSource;

        uint32_t valId = codeGenInfo.exchangePool.size();
        codeGenInfo.exchangePool.push_back(valInfo);
        writerToExchangeValId[edgeSource] = valId;


        // Create ExchangeWrite for srcRegion
        PartitioningGraphNodeProperty vp;
        vp.op = nullptr;
        vp.weight = 1;
        vp.exchangeValId = valId;
        vp.toucanOpName = CGToucanOPName::ExchangeWrite;

        auto exchangeWriteVtxId = boost::add_vertex(vp, regionGraphs[srcRegion]);
        boost::add_edge(srcNewId, exchangeWriteVtxId, regionGraphs[srcRegion]);
        numExchangeWrite++;
        exgWriteInRegion[srcRegion].push_back(exchangeWriteVtxId);
        // numExgWriteInRegion[srcRegion]++;
      }

      auto exchangeValId = writerToExchangeValId[edgeSource];

      if (!exchangeValIdToReader[dstRegion].contains(exchangeValId)) {
        // A new value that never read
        // Create ExchangeRead
        PartitioningGraphNodeProperty vp;
        vp.op = nullptr;
        vp.weight = 1;
        vp.exchangeValId = exchangeValId;
        vp.toucanOpName = CGToucanOPName::ExchangeRead;

        auto exchangeReadVtxId = boost::add_vertex(vp, regionGraphs[dstRegion]);
        boost::add_edge(exchangeReadVtxId, dstNewId, regionGraphs[dstRegion]);
        // update reader
        codeGenInfo.exchangePool[exchangeValId].readerIds.push_back(std::make_tuple(dstRegion, dstNewId));
        exchangeValIdToReader[dstRegion][exchangeValId] = exchangeReadVtxId;
        numExchangeRead++;
        exgReadInRegion[dstRegion].push_back(exchangeReadVtxId);
        // numExgReadInRegion[dstRegion]++;
      } else {
        auto exchangeReadVtxId = exchangeValIdToReader[dstRegion][exchangeValId];
        boost::add_edge(exchangeReadVtxId, dstNewId, regionGraphs[dstRegion]);
      }
    }
  }

  assert(codeGenInfo.exchangePool.size() == writerToExchangeValId.size());

  llvm::outs() << "Add " << codeGenInfo.exchangePool.size() << " exchange values, " << numExchangeWrite << " ExchangeWrite and " << numExchangeRead << " ExchangeRead\n";

  for (size_t rid = 0; rid < numRegions; rid++) {
    llvm::outs() << "Region " << rid << " has " << exgWriteInRegion[rid].size() << " ExchangeWrite and " << exgReadInRegion[rid].size() << " ExchangeRead\n";
  }

  for (size_t rid = 0; rid < numRegions; rid++) {
    llvm::outs() << "Region " << rid << " has " << boost::num_vertices(regionGraphs[rid]) << " vertices and " << boost::num_edges(regionGraphs[rid]) << " edges\n";
  }
  
  return;
}

void MultiRegionScheduler::levelizeAllPartitions(mlir::MLIRContext *context) {
  assert(!regionPartitions.empty());
  assert(regionPartLevels.empty());

  auto numRegions = regionGraphs.size();
  
  mlir::SmallVector<uint32_t> regionIds(numRegions);
  std::iota(regionIds.begin(), regionIds.end(), 0);

  for (auto regionId = 0; regionId < numRegions; regionId++) {
    regionPartLevels.emplace_back();
    for (uint32_t partId = 0; partId < regionPartitions[regionId].size(); partId++) {
      regionPartLevels.back().emplace_back();
    }
  }


  auto ret = mlir::failableParallelForEach(context, regionIds.begin(), regionIds.end(), [&](uint32_t regionId) {

    // levelize region graph
    mlir::SmallVector<mlir::SmallVector<uint32_t>> regionLevels;
    levelizeWorker(regionGraphs[regionId], regionLevels);
    auto &currentRegionPartitions = regionPartitions[regionId];
    auto &currentRegionPartLevels = regionPartLevels[regionId];

    uint32_t numPartitions = currentRegionPartitions.size();
    uint32_t regionNumVtxes = boost::num_vertices(regionGraphs[regionId]);

    mlir::SmallVector<mlir::SmallVector<uint32_t, 1>> vtxIdToPartIds;
    vtxIdToPartIds.resize(regionNumVtxes, {});

    // reserve space for each partitions
    currentRegionPartLevels.resize(numPartitions);

    for (uint32_t partId = 0; partId < numPartitions; partId++) {
      // maintain map from vtx id to partition ids
      for (auto &vtx: currentRegionPartitions[partId]) {
        vtxIdToPartIds[vtx].push_back(partId);
      }
      // reserve space for all levels
      currentRegionPartLevels[partId].resize(regionLevels.size());
    }


    // levelize
    for (uint32_t levelId = 0; levelId < regionLevels.size(); levelId++) {
      for (auto vtx: regionLevels[levelId]) {
        for (auto partId: vtxIdToPartIds[vtx]) {
          currentRegionPartLevels[partId][levelId].push_back(vtx);
        }
      }
    }

    // remove empty layers
    for (auto &eachPart: currentRegionPartLevels) {
      eachPart.erase(std::remove_if(eachPart.begin(), eachPart.end(), [](auto levelVec) { return levelVec.empty(); }), eachPart.end());
    }

    return success();
  });

  
  // debug: report
  bool printLevelStat = false;

  if (printLevelStat) {
    for (auto regionId = 0; regionId < numRegions; regionId++) {
      llvm::dbgs() << "Region " << regionId << "\n";
      for (uint32_t partId = 0; partId < regionPartitions[regionId].size(); partId++) {
        llvm::dbgs() << "  Partition " << partId << "\n";
        auto &currentPartLevels = regionPartLevels[regionId][partId];
        for (size_t levelId = 0; levelId < currentPartLevels.size(); levelId++) {
          uint32_t levelWeight = 0;
          for (auto vtx: currentPartLevels[levelId]) {
            levelWeight += regionGraphs[regionId][vtx].weight;
          }
          llvm::dbgs() << "    Level " << levelId << " has size of " << currentPartLevels[levelId].size() << ", weight of " << levelWeight << "\n";
        }
      }
    }
  }

  return;
}
