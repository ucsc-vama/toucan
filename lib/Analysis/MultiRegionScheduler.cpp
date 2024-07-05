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
    regionNewIdToVtxId.emplace_back();
    regionGraphs.emplace_back();
    auto &rg = regionGraphs.back();
    for (const auto &ev: eachRegionVtxes) {
      auto newVertex = boost::add_vertex(graph.g[ev], rg);

      assert(vtxIdToNewId[ev] == UINT32_MAX);
      assert(vtxIdToRegionId[ev] == UINT32_MAX);
      vtxIdToNewId[ev] = newVertex;
      vtxIdToRegionId[ev] = regionId;

      assert(newVertex == regionNewIdToVtxId.back().size());
      regionNewIdToVtxId.back().push_back(ev);
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
        valInfo.isPadding = false;
        valInfo.writerId = edgeSource;
        valInfo.writerRegionId = srcRegion;

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

        assert(exchangeWriteVtxId == regionNewIdToVtxId[srcRegion].size());
        regionNewIdToVtxId[srcRegion].push_back(UINT32_MAX);

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

        assert(exchangeReadVtxId == regionNewIdToVtxId[dstRegion].size());
        regionNewIdToVtxId[dstRegion].push_back(UINT32_MAX);

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

  for (size_t regionId = 0; regionId < numRegions; regionId++) {
    regionPartLevels.emplace_back();
    for (uint32_t partId = 0; partId < regionPartitions[regionId].size(); partId++) {
      regionPartLevels.back().emplace_back();
    }
  }

  auto ret = mlir::failableParallelForEach(context, regionIds.begin(), regionIds.end(), [&](uint32_t regionId) {
    // levelize region graph
    mlir::SmallVector<mlir::SmallVector<uint32_t>> regionLevels;
    levelizeWorker(regionGraphs[regionId], regionLevels);

    // assertions
    if (regionId != 0) {
      // has ExchangeRead, then the first level must be exchange reads
      for (auto vtx: regionLevels[0]) {
        assert(regionGraphs[regionId][vtx].toucanOpName == CGToucanOPName::ExchangeRead);
      }
    }
    if (regionId != numRegions - 1) {
      // has ExchangeWrite, then the last level must be exchange writes
      for (auto vtx: regionLevels.back()) {
        assert(regionGraphs[regionId][vtx].toucanOpName == CGToucanOPName::ExchangeWrite);
      }
    }
  
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
    for (size_t regionId = 0; regionId < numRegions; regionId++) {
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



void MultiRegionScheduler::sortRegistersForLocality(const PartitioningGraph &graph,  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regOrdered) {
  regOrdered.clear();

  mlir::SmallVector<mlir::SmallVector<uint32_t>> partToRegReads;
  mlir::SmallVector<mlir::SmallVector<uint32_t>> partToRegWrites;
  // Collect all reg reads (only appears at first level of region 0)
  for (size_t partId = 0; partId < regionPartLevels[0].size(); partId++) {
    assert(regionPartLevels[0][partId].size() >= 3);
    const auto & currentPartFirstLevel = regionPartLevels[0][partId][0];
    partToRegReads.emplace_back();

    for (auto newVtxId: currentPartFirstLevel) {
      auto mainGraphVtxId = regionNewIdToVtxId[0][newVtxId];
      // Note: ConstDecl may also present in first level
      if (graph[mainGraphVtxId].toucanOpName == CGToucanOPName::RegRead) {
        partToRegReads.back().push_back(mainGraphVtxId);
      }
    }
  }
  // Collect all reg writes (last level of last region)
  for (size_t partId = 0; partId < regionPartLevels.back().size(); partId++) {
    assert(regionPartLevels.back()[partId].size() >= 3);
    const auto & currentPartLastLevel = regionPartLevels.back()[partId].back();
    partToRegWrites.emplace_back();

    for (auto newVtxId: currentPartLastLevel) {
      auto mainGraphVtxId = regionNewIdToVtxId.back()[newVtxId];
      if (graph[mainGraphVtxId].toucanOpName == CGToucanOPName::RegWrite) {
        partToRegWrites.back().push_back(mainGraphVtxId);
      }
    }
    assert(!partToRegWrites.back().empty());
  }


  // Segment by writer, then reader
  // Sort by reader: shared read, read by p0, p1, p2, ...

  // Here we assume replication rate is relatively small
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> replicatedRegReadVals;
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> regReadValToPartId;
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regValRead, regValWrite;

  for (size_t partId = 0; partId < partToRegReads.size(); partId++) {
    for (auto vtxId: partToRegReads[partId]) {
      auto regReadOp = cast<toucan::RegReadOp>(graph[vtxId].op);
      auto regVal = regReadOp.getReg();
      regValRead.insert(regVal);

      if (!replicatedRegReadVals.contains(regVal)) {
        if (regReadValToPartId.contains(regVal)) {
          // has at least 2 writer
          replicatedRegReadVals.insert(regVal);
          regReadValToPartId.erase(regVal);
        } else {
          regReadValToPartId[regVal] = partId;
        }
      }
    }
  }

  // reg vals that read by multiple partitions (multiple reader)
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> sharedVals;
  // reg vals that has only 1 reader
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> groupedVals;
  // reg vals with no reader
  // mlir::SmallVector<mlir::TypedValue<toucan::RegType>> readOnlyRegVals;
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> writeOnlyRegVals;

  // auto numReadParts = partToRegReads.size();
  groupedVals.resize(partToRegReads.size());

  for (auto &eachPartWrites: partToRegWrites) {
    sharedVals.clear();
    for (size_t partId = 0; partId < partToRegReads.size(); partId++) {
      groupedVals[partId].clear();
    }

    for (auto &vtxId: eachPartWrites) {
      auto regWriteOp = cast<toucan::RegWriteOp>(graph[vtxId].op);
      auto regVal = regWriteOp.getReg();
      regValWrite.insert(regVal);

      if (!regReadValToPartId.contains(regVal)) {
        writeOnlyRegVals.push_back(regVal);
      } else if (replicatedRegReadVals.contains(regVal)) {
        sharedVals.push_back(regVal);
      } else {
        assert(regReadValToPartId.contains(regVal));
        auto partId = regReadValToPartId[regVal];
        // assert(partId < numReadParts);
        groupedVals[partId].push_back(regVal);
      }
    }

    // put shareVals at the begining, then group by reader partition ID
    regOrdered.emplace_back();
    std::copy(sharedVals.begin(), sharedVals.end(), std::back_inserter(regOrdered.back()));
    for (auto &eachGroupVals: groupedVals) {
      std::copy(eachGroupVals.begin(), eachGroupVals.end(), std::back_inserter(regOrdered.back()));
    }
    std::copy(writeOnlyRegVals.begin(), writeOnlyRegVals.end(), std::back_inserter(regOrdered.back()));
    // std::copy(readOnlyRegVals.begin(), readOnlyRegVals.end(), std::back_inserter(regOrdered.back()));
  }

  // Read only vals. Push them to last section.
  uint32_t numReadOnlyVals = 0;
  for (auto eachVal: regValRead) {
    if (!regValWrite.contains(eachVal)) {
      regOrdered.back().push_back(eachVal);
      numReadOnlyVals ++;
    }
  }

  llvm::dbgs() << "Found " << numReadOnlyVals << " read only values\n";

  return;
}


void MultiRegionScheduler::sortOpsAndExchangeValsForLocality(const mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> &regPoolOrdered, mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> &exchangeValIdOrdered) {
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> regValToOrder;
  mlir::SmallVector<uint32_t> vtxIdToLevelOrder;
  mlir::SmallVector<uint32_t> exchangeValIdToOrder;

  uint32_t exchangeValIdOrderNext = 0;
  exchangeValIdToOrder.resize(codeGenInfo.exchangePool.size(), UINT32_MAX);
  
  // Temporaries
  mlir::SmallVector<uint32_t> exchangeValShared;
  mlir::SmallVector<mlir::SmallVector<uint32_t>> exchangeValUnique;
  // only used for first level register reads
  mlir::DenseMap<uint32_t, uint32_t> vtxIdToOrder;
  // used for every level, sorting cache
  mlir::DenseMap<uint32_t, uint32_t> currentLevelVtxSortKey;
  mlir::SmallVector<uint32_t> allInVtxOrder;

  uint32_t regOrder = 0;
  for (const auto &eachSection: regPoolOrdered) {
    for (const auto &eachVal: eachSection) {
      regValToOrder[eachVal] = regOrder;
      regOrder++;
    }
  }

  auto numRegions = regionGraphs.size();
  // uint32_t totalParts = 0;
  // for (const auto &er: regionPartLevels) {
  //   totalParts += er.size();
  // }

  for (size_t regionId = 0; regionId < regionGraphs.size(); regionId++) {
    auto numParts = regionPartLevels[regionId].size();
    const auto &currentRegionGraph = regionGraphs[regionId];
    // llvm::dbgs() <<"Working on region " << regionId << "\n";

    for (size_t partId = 0; partId < regionPartLevels[regionId].size(); partId++) {
      // llvm::dbgs() <<"Working on part " << partId << "\n";

      auto numLevels = regionPartLevels[regionId][partId].size();
      vtxIdToLevelOrder.resize(boost::num_vertices(currentRegionGraph), UINT32_MAX);

      uint32_t levelOrder = 0;


      for (size_t levelId = 0; levelId < regionPartLevels[regionId][partId].size(); levelId++) {

        // llvm::dbgs() <<"Working on level " << levelId << "\n";

        auto &currentLevelVtxes = regionPartLevels[regionId][partId][levelId];

        if (levelId == 0) {
          // top level
          if (regionId == 0) {
            // reg reads. Sort by reg order
            vtxIdToOrder.clear();
            vtxIdToOrder.reserve(currentLevelVtxes.size());
            for (auto eachVtx: currentLevelVtxes) {
              if (currentRegionGraph[eachVtx].toucanOpName == CGToucanOPName::RegRead) {
                auto regReadOp = cast<toucan::RegReadOp>(currentRegionGraph[eachVtx].op);
                auto regVal = regReadOp.getReg();
                vtxIdToOrder[eachVtx] = regValToOrder[regVal];
              } else {
                // Maybe a constDecl
                assert(currentRegionGraph[eachVtx].toucanOpName == CGToucanOPName::ConstDecl);
                // ConstDecl will merged to valuePool after scheduling. For now simply move them to last
                vtxIdToOrder[eachVtx] = UINT32_MAX;
              }
            }

            std::sort(currentLevelVtxes.begin(), currentLevelVtxes.end(), [&vtxIdToOrder](uint32_t a, uint32_t b) {
              return vtxIdToOrder[a] < vtxIdToOrder[b];
            });
          } else {
            // exchange reads
            // order for exchange read
            // order by exchangeValIdToOrder
            vtxIdToOrder.clear();
            for (auto eachVtx: currentLevelVtxes) {
              auto exchangeValId = currentRegionGraph[eachVtx].exchangeValId;
              vtxIdToOrder[eachVtx] = exchangeValIdToOrder[exchangeValId];
            }

            std::sort(currentLevelVtxes.begin(), currentLevelVtxes.end(), [&](const uint32_t a, const uint32_t b) {
              return vtxIdToOrder[a] < vtxIdToOrder[b];
            });
            
          }
        } else {
          
          // levelId != 0
          // Not first level, sort by last reference

          // Find order of last inNeight
          currentLevelVtxSortKey.clear();
          for (auto eachVtx: currentLevelVtxes) {
            allInVtxOrder.clear();
            auto in_edges_range = boost::in_edges(eachVtx, currentRegionGraph);
            for (auto ei = in_edges_range.first; ei != in_edges_range.second; ++ei) {
              auto srcVtx = boost::source(*ei, currentRegionGraph);
              auto srcOrder = vtxIdToLevelOrder[srcVtx];
              assert(srcOrder != UINT32_MAX);

              allInVtxOrder.push_back(srcOrder);
            }
            assert(!allInVtxOrder.empty());
            auto maxOrder = *std::max_element(allInVtxOrder.begin(), allInVtxOrder.end());
            currentLevelVtxSortKey[eachVtx] = maxOrder;
          }

          std::sort(currentLevelVtxes.begin(), currentLevelVtxes.end(), [&](const uint32_t a, const uint32_t b) {
            return currentLevelVtxSortKey[a] < currentLevelVtxSortKey[b];
          });
        }

        // update level order. This will be used by following levels
        for (auto vtx: currentLevelVtxes) {
          vtxIdToLevelOrder[vtx] = levelOrder;
          levelOrder++;
        }

      }

      // Sort exchange write vals
      if ((regionId != (numRegions - 1))) {
        // llvm::dbgs() << "Working on ExchangeWrites\n";
        // Last level ExchangeWrite
        // val used by more than 1 partitons, or cross region
        exchangeValShared.clear();
        // val used by only 1 part
        exchangeValUnique.clear();
        exchangeValUnique.resize(numParts);

        if (partId == 0) {
          exchangeValIdOrdered.emplace_back();
        }
        exchangeValIdOrdered.back().emplace_back();
        for (auto vtxId: regionPartLevels[regionId][partId].back()) {
          assert(currentRegionGraph[vtxId].toucanOpName == CGToucanOPName::ExchangeWrite);

          auto exchangeValId = currentRegionGraph[vtxId].exchangeValId;

          assert(codeGenInfo.exchangePool[exchangeValId].readerIds.size() != 0);

          if (codeGenInfo.exchangePool[exchangeValId].readerIds.size() > 1) {
            // a shared value
            exchangeValShared.push_back(exchangeValId);
          } else {
            auto readerRegionId = std::get<0>(codeGenInfo.exchangePool[exchangeValId].readerIds.back());
            auto readerVtxId = std::get<1>(codeGenInfo.exchangePool[exchangeValId].readerIds.back());
            if (readerRegionId == regionId + 1) {
              // used by next region
              auto userPartId = regionNewIdToPartId[regionId+1][readerVtxId];
              exchangeValUnique[userPartId].push_back(exchangeValId);
            } else {
              // A cross region read (uncommon). treat as shared
              exchangeValShared.push_back(exchangeValId);
            }
          }
        }
        // merge all together
        std::copy(exchangeValShared.begin(), exchangeValShared.end(), std::back_inserter(exchangeValIdOrdered.back().back()));
        for (auto &eachSection: exchangeValUnique) {
          if (!eachSection.empty()) {
            std::copy(eachSection.begin(), eachSection.end(), std::back_inserter(exchangeValIdOrdered.back().back()));
          }
        }

        // save order for internal use
        for (auto ev: exchangeValIdOrdered.back().back()) {
          exchangeValIdToOrder[ev] = exchangeValIdOrderNext;
          exchangeValIdOrderNext++;
        }
      }
    }
  }
}

void MultiRegionScheduler::generateRegMemLayout(DesignGraph &graph) {
  // collect all reg and memory, generate layout
  codeGenInfo.regPool.clear();
  codeGenInfo.memPool.clear();

  assert(regionPartLevels.size() > 0);

  auto numRegions = regionGraphs.size();

  // regionNewIdToPartId
  regionNewIdToPartId.resize(numRegions);
  for (uint32_t regionId = 0; regionId < numRegions; regionId++) {
    regionNewIdToPartId[regionId].resize(boost::num_vertices(regionGraphs[regionId]), UINT32_MAX);
    for (size_t partId = 0; partId < regionPartLevels[regionId].size(); partId++) {
      for (const auto &eachLevelVtxes: regionPartLevels[regionId][partId]) {
        for (auto eachVtx: eachLevelVtxes) {
          regionNewIdToPartId[regionId][eachVtx] = partId;
        }
      }
    }
  }


  // Writer part -> val. Needs padding
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered;
  // order to exchangeValId
  // region -> writer part -> valId. Needs padding
  mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> exchangeValIdOrdered;

  // llvm::dbgs() << "Sorting registers\n";
  sortRegistersForLocality(graph.g, regPoolOrdered);
  // llvm::dbgs() << "Sorting ops and exchange vals\n";
  sortOpsAndExchangeValsForLocality(regPoolOrdered, exchangeValIdOrdered);


  // Now, registers and exchangeVal location and ops are sorted
  // Allocate space

  // Allocate storage for all registers
  for (auto &eachSection: regPoolOrdered) {
    for (auto &regVal: eachSection) {
      // allocate storate for every register
      auto regDefiningOp = regVal.getDefiningOp();

      CGRegMetaInfo regMeta;

      regMeta.namehint = getSVNameHintAttr(regDefiningOp);
      auto fragmentIdAttr = getSignalFragmentIDAttr(regDefiningOp);
      if (fragmentIdAttr) {
        regMeta.fragment_id = fragmentIdAttr->getInt();
      } else {
        regMeta.fragment_id = UINT32_MAX;
      }
      regMeta.bitWidth = regVal.getType().getElementWidth();
      regMeta.isPadding = false;
      regMeta.isIO = hasIOSignalMarker(regDefiningOp);

      auto regId = codeGenInfo.regPool.size();
      codeGenInfo.regPool.push_back(regMeta);
      codeGenInfo.toucanRegToId[regVal] = regId;
    }
    // add padding regs
    for (size_t i = 0; i < partitionPaddingSpace; i++) {
      CGRegMetaInfo paddingRegMeta;

      paddingRegMeta.isPadding = true;
      paddingRegMeta.bitWidth = 0;
      paddingRegMeta.fragment_id = 0;
      paddingRegMeta.isIO = false;

      // auto regId = codeGenInfo.regPool.size();
      codeGenInfo.regPool.push_back(paddingRegMeta);
    }
  }
  
  // collect all reg and memory, generate layout
  // Here we assume each memory has at least 1 writer
  uint64_t memBaseAddr = 0;
  for (size_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    // for each mem write
    auto vtxOpName = graph.g[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::MemWrite) {
      // Note: For now, mems with multiple write ports are still merged, so at this time, each memory will only have 1 writer.
      auto memWriteOp = cast<toucan::MemWriteOp>(graph.g[vtxId].op);
      auto memVal = memWriteOp.getMem();
      auto memDefiningOp = memVal.getDefiningOp();

      CGMemMetaInfo memMeta;

      memMeta.namehint = getSVNameHintAttr(memDefiningOp);
      auto fragmentIdAttr = getSignalFragmentIDAttr(memDefiningOp);
      if (fragmentIdAttr) {
        memMeta.fragment_id = fragmentIdAttr->getInt();
      } else {
        memMeta.fragment_id = UINT32_MAX;
      }
      memMeta.bitWidth = memVal.getType().getElementWidth();
      memMeta.memDepth = memVal.getType().getDepth();
      memMeta.hasMultipleWriter = (graph.g[vtxId].weight > 1);

      // if a memory has multiple writer, add extra padding to avoid possible write conflict
      assert(memMeta.bitWidth <= 4);
      uint64_t memCapacity = (memMeta.hasMultipleWriter) ? memMeta.memDepth * multiWriterMemElemBytes : memMeta.memDepth;
      memMeta.memBase = memBaseAddr;
      memBaseAddr += (memCapacity + memPaddingSpace);

      auto memId = codeGenInfo.memPool.size();
      codeGenInfo.memPool.push_back(memMeta);
      codeGenInfo.toucanMemToId[memVal] = memId;
    }
  }
  codeGenInfo.totalMemSize = memBaseAddr;




  // Reorder exchangePool
  // old Id -> new Id
  mlir::SmallVector<uint32_t> exchangePoolReorderIdMap;
  uint32_t expectedExchangePoolSize = 0;
  for (const auto &eachRegion: exchangeValIdOrdered) {
    for (const auto &eachWriterPart: eachRegion) {
      expectedExchangePoolSize += eachWriterPart.size();
      expectedExchangePoolSize += partitionPaddingSpace;
    }
  }
  exchangePoolReorderIdMap.resize(expectedExchangePoolSize, UINT32_MAX);

  // region -> writer part -> valId. Needs padding
  uint32_t nextValId = 0;
  for (const auto &eachRegion: exchangeValIdOrdered) {
    for (const auto &eachWriterPart: eachRegion) {
      for (const auto eachExchangeValId: eachWriterPart) {
        exchangePoolReorderIdMap[eachExchangeValId] = nextValId;
        nextValId++;
      }
      // add padding
      for (size_t i = 0; i < partitionPaddingSpace; i++) {
        CGExchangeValueMetaInfo paddingValMeta;
        paddingValMeta.isPadding = true;
        paddingValMeta.writerId = UINT32_MAX;
        paddingValMeta.writerRegionId = UINT32_MAX;

        uint32_t paddingValOldId = codeGenInfo.exchangePool.size();
        codeGenInfo.exchangePool.push_back(paddingValMeta);
        exchangePoolReorderIdMap[paddingValOldId] = nextValId;
        nextValId++;
      }
    }
  }
  assert(nextValId == expectedExchangePoolSize);

  // sort exchangePool
  mlir::SmallVector<CGExchangeValueMetaInfo> exchangePoolOrdered;
  exchangePoolOrdered.resize(codeGenInfo.exchangePool.size());
  for (uint32_t oldId = 0; oldId < exchangePoolReorderIdMap.size(); oldId++) {
    auto newId = exchangePoolReorderIdMap[oldId];
    assert(newId != UINT32_MAX);
    exchangePoolOrdered[newId] = codeGenInfo.exchangePool[oldId];
  }
  std::swap(exchangePoolOrdered, codeGenInfo.exchangePool);

  // modify all references
  for (auto &eachRegionGraph: regionGraphs) {
    for (uint32_t vtxId = 0; vtxId < boost::num_vertices(eachRegionGraph); vtxId++) {
      auto opName = eachRegionGraph[vtxId].toucanOpName;
      if ((opName == CGToucanOPName::ExchangeRead) || (opName == CGToucanOPName::ExchangeWrite)) {
        auto oldValId = eachRegionGraph[vtxId].exchangeValId;
        auto newValId = exchangePoolReorderIdMap[oldValId];
        eachRegionGraph[vtxId].exchangeValId = newValId;
      }
    }
  }


  return;
}

// Scheduler entry point
void MultiRegionScheduler::schedule(DesignGraph &graph) {

  auto numRegions = regionGraphs.size();

  generateRegMemLayout(graph);


  // dedup strings
  collectPrintString(graph, codeGenInfo.printStrings);
  return;
}
