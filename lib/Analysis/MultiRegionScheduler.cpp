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
#include <algorithm>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;




void MultiRegionScheduler::levelizeGraphForCut(DesignGraph &graph) {
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

  mlir::DenseMap<mlir::Operation*, uint32_t> opToVtxId;
  opToVtxId.reserve(graphSize);


  size_t currentRegionId = 0;
  auto nextCutPoint = cutPoints[0];
  // VecDecl and VecRead must be in same region.
  // VecDecl and VecRead shall exist in adjacent levels
  mlir::DenseSet<uint32_t> vecReadsInNextRegion;
  for (size_t levelId = 0; levelId < graphLevels.size(); levelId++) {
    const auto currentLevel = graphLevels[levelId];
    // TODO: VecDecl and VecRead must be in same region!
    bool isLastLevelInRegion = levelId > nextCutPoint;

    if (isLastLevelInRegion) {
      currentRegionId++;
      // regionVtxes.emplace_back();
      nextCutPoint = (currentRegionId >= cutPoints.size()) ? graphLevels.size() : cutPoints[currentRegionId];
    }
    
    for (auto ev: currentLevel) {
      vtxIdToRegionId[ev] = currentRegionId;
    }
    // regionVtxes.back().insert(regionVtxes.back().end(), currentLevel.begin(), currentLevel.end());
    // assert(currentRegionId + 1 == regionVtxes.size());
    // llvm::dbgs() << "Put level " << levelId << " to region " << currentRegionId << "\n";
  }
  regionVtxes.resize(currentRegionId + 1);

  // Move VecDecl to its user region (backwards)
  mlir::SmallVector<uint32_t> vecUserRegions;
  for (uint32_t vtxId = 0; vtxId < graphSize; vtxId++) {
    auto tOpName = graph.g[vtxId].toucanOpName;
    auto opPtr = graph.g[vtxId].op;
    assert(opPtr != nullptr);
    assert(!opToVtxId.contains(opPtr));
    opToVtxId[opPtr] = vtxId;

    if (tOpName == CGToucanOPName::VecDecl) {
      auto vecDeclRegion = vtxIdToRegionId[vtxId];

      vecUserRegions.clear();
      auto vecUserEdges = boost::out_edges(vtxId, graph.g);
      for (auto ei = vecUserEdges.first; ei != vecUserEdges.second; ei++) {
        auto userVtxId = boost::target(*ei, graph.g);
        auto userVtxRegion = vtxIdToRegionId[userVtxId];
        vecUserRegions.push_back(userVtxRegion);
      }

      assert(!vecUserRegions.empty());
      auto vecUserRegion = vecUserRegions[0];
      for (auto eachUserRegion: vecUserRegions) {
        assert(vecUserRegion == eachUserRegion && "All VecDecl users should be in same region!");
      }

      if (vecDeclRegion != vecUserRegion) {
        assert(vecUserRegion > vecDeclRegion);
        // move VecDecl to user region
        // llvm::dbgs() << "Move vtx " << vtxId << " from region " << vecDeclRegion << " to region " << vecUserRegion << "\n";
        vtxIdToRegionId[vtxId] = vecUserRegion;
      }
    }
  }

  for (uint32_t vtxId = 0; vtxId < graphSize; vtxId++) {
    auto regionId = vtxIdToRegionId[vtxId];
    assert(regionId != UINT32_MAX);
    regionVtxes[regionId].push_back(vtxId);
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
      // assert(vtxIdToRegionId[ev] == UINT32_MAX);
      vtxIdToNewId[ev] = newVertex;
      // vtxIdToRegionId[ev] = regionId;

      assert(newVertex == regionNewIdToVtxId.back().size());
      regionNewIdToVtxId.back().push_back(ev);

      // if (graph.g[ev].toucanOpName == CGToucanOPName::VecDecl) {
      //   int numInputs = cast<toucan::DefVectorOp>(graph.g[ev].op).getInputs().size();
      //   auto inEdgeRange = boost::in_edges(ev, graph.g);
      //   auto inEdges = std::distance(inEdgeRange.first, inEdgeRange.second);
      //   if (numInputs != inEdges) {

      //   llvm::dbgs() << "VecDecl has " << numInputs << " IR inputs, " << inEdges << " in edges\n";
      //   }
      //   assert(numInputs == inEdges);
      // }
    }

    llvm::outs() << "Before add extra IO, region " << regionId << " has " << boost::num_vertices(rg) << " verticies\n";

    regionId++;
  }


  // Add edges
  auto numRegions = regionGraphs.size();

  mlir::DenseMap<uint32_t, uint32_t> writerToExchangeValId;
  mlir::DenseSet<uint32_t> processedVecDeclVtxes;
  mlir::SmallVector<mlir::DenseMap<uint32_t, uint32_t>> exchangeValIdToReader(numRegions);
  uint32_t numExchangeWrite = 0;
  uint32_t numExchangeRead = 0;
  // mlir::SmallVector<uint32_t> numExgWriteInRegion(numRegions);
  // mlir::SmallVector<uint32_t> numExgReadInRegion(numRegions);

  mlir::SmallVector<mlir::SmallVector<uint32_t>> exgWriteInRegion(numRegions);
  mlir::SmallVector<mlir::SmallVector<uint32_t>> exgReadInRegion(numRegions);
          
  mlir::DenseMap<mlir::Value, uint32_t> vecReadValToExchangeId;
  mlir::SmallVector<uint32_t> vecReadValUsersInOtherRegion;

  for (size_t i = 0; i < numRegions; i++) {
    exgWriteInRegion.emplace_back();
    exgReadInRegion.emplace_back();
  }

  auto opResultValUsedByOtherRegion = [&](const mlir::Value &val, uint32_t currentRegion) {
    for (const auto &eachUserOp: val.getUsers()) {
      assert(opToVtxId.contains(eachUserOp));
      auto userVtxId = opToVtxId[eachUserOp];
      auto userRegionId = vtxIdToRegionId[userVtxId];
      if (userRegionId != currentRegion) return true;
    }
    return false;
  };

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

      auto srcOpName = graph.g[edgeSource].toucanOpName;
      auto srcWeight = graph.g[edgeSource].weight;

      if (srcWeight == 1) {
        // unmerged vtx: lut, mem, etc, and vecDecl with only 1 user
        if (!writerToExchangeValId.contains(edgeSource)) {
          // Allocate a new exchange val, if it's not already exist
          CGExchangeValueMetaInfo valInfo;
          valInfo.isPadding = false;
          auto writerOp = graph.g[edgeSource].op;
          assert(writerOp->getNumResults() == 1);
          valInfo.val = writerOp->getResult(0);
          valInfo.writerId = srcNewId;
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
          // boost::add_edge(exchangeReadVtxId, dstNewId, regionGraphs[dstRegion]);

          assert(exchangeReadVtxId == regionNewIdToVtxId[dstRegion].size());
          regionNewIdToVtxId[dstRegion].push_back(UINT32_MAX);

          // update reader
          codeGenInfo.exchangePool[exchangeValId].readerIds.push_back(std::make_tuple(dstRegion, exchangeReadVtxId));
          exchangeValIdToReader[dstRegion][exchangeValId] = exchangeReadVtxId;
          numExchangeRead++;
          exgReadInRegion[dstRegion].push_back(exchangeReadVtxId);
          // numExgReadInRegion[dstRegion]++;
        }
        auto exchangeReadVtxId = exchangeValIdToReader[dstRegion][exchangeValId];
        assert(regionGraphs[dstRegion][exchangeReadVtxId].toucanOpName == CGToucanOPName::ExchangeRead);
        assert(exchangeReadVtxId < boost::num_vertices(regionGraphs[dstRegion]));
        assert(dstNewId < boost::num_vertices(regionGraphs[dstRegion]));
        boost::add_edge(exchangeReadVtxId, dstNewId, regionGraphs[dstRegion]);
      } else {
        // Even though it's possible for a VecDecl to have weight more than 1,
        // VecDecl and its user should not cross region boundary
        // Thus VecDecl should NOT appear here
        assert(srcOpName != CGToucanOPName::VecDecl);
        // MemWrite should not cross region (and only appears in last region, last level)
        // MemRead exclusively owns a vector
        // The only possible case is VecRead
        assert(srcOpName == CGToucanOPName::VecRead);


        // Note: this process partially expands VecReads, thus only need to run once for each VecDecl
        // After this process, VecReads are still merged, but users in another region is splitted: edges are connected to different ExchangeReads
        if (!processedVecDeclVtxes.contains(edgeSource)) {
          processedVecDeclVtxes.insert(edgeSource);
                  
          assert(!writerToExchangeValId.contains(edgeSource));

          auto vecReadOp = cast<toucan::VectorReadOp>(graph.g[edgeSource].op);
          auto vecHandle = vecReadOp.getHandle();
          auto vecUsers = vecHandle.getUsers();

          assert(llvm::range_size(vecUsers) == srcWeight);

          vecReadValToExchangeId.clear();
          vecReadValUsersInOtherRegion.clear();

          auto out_edge_range = boost::out_edges(edgeSource, graph.g);
          for (auto ei = out_edge_range.first; ei != out_edge_range.second; ei++) {
            auto vecReadResultUserVtx = boost::target(*ei, graph.g);
            auto vecReadResultUserRegion = vtxIdToRegionId[vecReadResultUserVtx];

            // If the user is in different region with VecRead, save it.
            if (vecReadResultUserRegion != srcRegion) {
              vecReadValUsersInOtherRegion.push_back(vecReadResultUserVtx);
            }
          }

          // Allocate an ExchangeVal slot if the VecRead result is used by other regions
          for (auto eachVecUser: vecUsers) {
            auto vecReadOp = cast<toucan::VectorReadOp>(eachVecUser);
            // llvm::dbgs() << "A new VecRead IR\n";
            // vecReadOp.print(llvm::dbgs());
            // llvm::dbgs() << "\n";
            // Note: Dont insert to writerToExchangeValId, as this vtx creates multiple exchange vals
            // Allocate exchange val
            auto resultVal = vecReadOp.getResult();

            if (!opResultValUsedByOtherRegion(resultVal, srcRegion)) continue;

            CGExchangeValueMetaInfo valInfo;
            valInfo.isPadding = false;
            valInfo.val = resultVal;
            valInfo.writerId = srcNewId;
            valInfo.writerRegionId = srcRegion;

            uint32_t valId = codeGenInfo.exchangePool.size();
            codeGenInfo.exchangePool.push_back(valInfo);
            assert(!vecReadValToExchangeId.contains(resultVal));
            vecReadValToExchangeId[resultVal] = valId;

            // Create ExchangeWrite for srcRegion
            PartitioningGraphNodeProperty vp;
            vp.op = nullptr;
            vp.weight = 1;
            vp.exchangeValId = valId;
            vp.toucanOpName = CGToucanOPName::ExchangeWrite;

            auto exchangeWriteVtxId = boost::add_vertex(vp, regionGraphs[srcRegion]);
            assert(srcNewId < boost::num_vertices(regionGraphs[srcRegion]));
            boost::add_edge(srcNewId, exchangeWriteVtxId, regionGraphs[srcRegion]);

            assert(exchangeWriteVtxId == regionNewIdToVtxId[srcRegion].size());
            regionNewIdToVtxId[srcRegion].push_back(UINT32_MAX);

            numExchangeWrite++;
            exgWriteInRegion[srcRegion].push_back(exchangeWriteVtxId);
          }

          // Create ExchangeRead for VecRead result users, if necessary
          // Also create edges.
          for (auto vecReadResultUserVtx: vecReadValUsersInOtherRegion) {
            auto vecReadResultUserRegion = vtxIdToRegionId[vecReadResultUserVtx];

            assert(vecReadResultUserRegion != srcRegion);

            auto readUserOp = graph.g[vecReadResultUserVtx].op;
            // llvm::dbgs() << " New VecRead result user\n";
            // readUserOp->print(llvm::dbgs());
            // llvm::dbgs() << "\n";
            auto inputVals = readUserOp->getOperands();
            for (auto eachInputVal: inputVals) {
              if (vecReadValToExchangeId.contains(eachInputVal)) {
                // llvm::dbgs() << " Read the vec result!\n";
                // It's a result of previous VecRead
                auto exchangeValId = vecReadValToExchangeId[eachInputVal];
                if (!exchangeValIdToReader[vecReadResultUserRegion].contains(exchangeValId)) {
                  // llvm::dbgs() << "Create new ExchangeReadVtx\n";
                  // A new value that never read
                  // Create ExchangeRead
                  PartitioningGraphNodeProperty vp;
                  vp.op = nullptr;
                  vp.weight = 1;
                  vp.exchangeValId = exchangeValId;
                  vp.toucanOpName = CGToucanOPName::ExchangeRead;

                  auto exchangeReadVtxId = boost::add_vertex(vp, regionGraphs[vecReadResultUserRegion]);

                  assert(exchangeReadVtxId == regionNewIdToVtxId[vecReadResultUserRegion].size());
                  regionNewIdToVtxId[vecReadResultUserRegion].push_back(UINT32_MAX);

                  // update reader
                  codeGenInfo.exchangePool[exchangeValId].readerIds.push_back(std::make_tuple(vecReadResultUserRegion, exchangeReadVtxId));
                  exchangeValIdToReader[vecReadResultUserRegion][exchangeValId] = exchangeReadVtxId;
                  numExchangeRead++;
                  exgReadInRegion[vecReadResultUserRegion].push_back(exchangeReadVtxId);
                  // numExgReadInRegion[dstRegion]++;
                }
                auto exchangeReadVtxId = exchangeValIdToReader[vecReadResultUserRegion][exchangeValId];
                assert(regionGraphs[vecReadResultUserRegion][exchangeReadVtxId].toucanOpName == CGToucanOPName::ExchangeRead);
                auto vecReadResultUserNewId = vtxIdToNewId[vecReadResultUserVtx];
                assert(vecReadResultUserNewId != UINT32_MAX);
                // llvm::dbgs() << "Add edge from " << exchangeReadVtxId << " to " << vecReadResultUserNewId << "\n";
                auto numRegionVertices = boost::num_vertices(regionGraphs[vecReadResultUserRegion]);
                assert(vecReadResultUserRegion > srcRegion);
                assert(exchangeReadVtxId < numRegionVertices);
                assert(vecReadResultUserNewId < numRegionVertices);
                boost::add_edge(exchangeReadVtxId, vecReadResultUserNewId, regionGraphs[vecReadResultUserRegion]);
              }
            }
          }

        }
      }

    }
  }

  llvm::outs() << "Add " << codeGenInfo.exchangePool.size() << " exchange values, " << numExchangeWrite << " ExchangeWrite and " << numExchangeRead << " ExchangeRead\n";

  for (size_t rid = 0; rid < numRegions; rid++) {
    llvm::outs() << "Region " << rid << " has " << exgWriteInRegion[rid].size() << " ExchangeWrite and " << exgReadInRegion[rid].size() << " ExchangeRead\n";
  }

  for (size_t rid = 0; rid < numRegions; rid++) {
    llvm::outs() << "Region " << rid << " has " << boost::num_vertices(regionGraphs[rid]) << " vertices and " << boost::num_edges(regionGraphs[rid]) << " edges\n";
  }
  
  return;
}

LogicalResult MultiRegionScheduler::levelizeAllPartitions(mlir::MLIRContext *context) {
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
    } else {
      for (auto vtx: regionLevels[0]) {
        assert(vtx != UINT32_MAX);
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
          assert(vtx != UINT32_MAX);
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

  if (failed(ret)) {
    llvm::errs() << "Fail to levelize graphs!\n";
    return ret;
  }

  
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

  return success();
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
      assert(mainGraphVtxId != UINT32_MAX);
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
  }

  // Read only vals. Push them to last section.
  // uint32_t numReadOnlyVals = 0;
  for (auto eachVal: regValRead) {
    if (!regValWrite.contains(eachVal)) {
      regOrdered.back().push_back(eachVal);
      // numReadOnlyVals ++;
    }
  }

  // llvm::dbgs() << "Found " << numReadOnlyVals << " read only values\n";

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

  for (size_t regionId = 0; regionId < regionGraphs.size(); regionId++) {
    auto numParts = regionPartLevels[regionId].size();
    const auto &currentRegionGraph = regionGraphs[regionId];
    // llvm::dbgs() <<"Working on region " << regionId << "\n";

    for (size_t partId = 0; partId < regionPartLevels[regionId].size(); partId++) {
      // llvm::dbgs() <<"Working on part " << partId << "\n";

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

  // 0. initialize regionNewIdToPartId
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


  // 1. sort registers and exchange vals.
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

  // 2. Allocate storage for all registers
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
  
  // 3. Allocate storage for all memories
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




  // 4. Reorder exchangePool
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


void MultiRegionScheduler::collectConstant(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> firstLevelOps) {
  // Collect all consts, populate value pool
  // ConstDecl only exists in first level
  for (auto vtxId: firstLevelOps) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::ConstDecl) {
      auto op = graph[vtxId].op;

      if (auto constOp = dyn_cast<toucan::ConstantOp>(op)) {
        // regular const
        auto constVal = constOp.getValue();
        auto bitWidth = constVal.getBitWidth();
        assert(bitWidth <= 4);
        auto rawVal = static_cast<uint8_t>(constVal.getZExtValue());
        // save op result value
        assert(rawVal == (rawVal & ((1 << bitWidth) - 1)));
        auto valId = partInfo.valuePool.size();

        auto constResultVal = constOp.getResult();
        assert(!partInfo.valueToValId.contains(constResultVal));
        partInfo.valueToValId[constResultVal] = valId;

        partInfo.valuePool.push_back({true, false, rawVal, op, 0, static_cast<uint8_t>(bitWidth), std::nullopt, 0});
      } else {
        // it must be a vector const
        auto defConstVecOp = cast<toucan::DefConstVectorOp>(op);
        // save op result value
        // Vec result map to first vec element
        auto vecHandle = defConstVecOp.getHandle();
        auto bitWidth = vecHandle.getType().getElementWidth();
        assert(bitWidth <= 4);

        auto valId = partInfo.valuePool.size();
        assert(!partInfo.valueToValId.contains(vecHandle));
        partInfo.valueToValId[vecHandle] = valId;

        // Why reverse? vector decl op elements are MSB first. Reorder to LSB first to make vecRead's life easier
        for (auto &vecValElem: llvm::reverse(defConstVecOp.getValues().getValue())) {
          auto elemVal = cast<mlir::IntegerAttr>(vecValElem).getValue();
          auto elemValWidth = elemVal.getBitWidth();
          assert(elemValWidth <= 4);

          auto elemValMask = static_cast<uint8_t>((1 << elemValWidth) - 1);
          uint8_t rawVal = elemValMask & static_cast<uint8_t>(elemVal.getZExtValue());
          partInfo.valuePool.push_back({true, false, rawVal, op, 0, static_cast<uint8_t>(bitWidth), std::nullopt, 0});
        }
      }
    }
  }
}

// Note: This step is same as SingleRegionScheduler
void MultiRegionScheduler::scheduleFirstLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &firstLevelOps) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;
  currentLevelOps.reserve(firstLevelOps.size());

  for (auto vtxId: firstLevelOps) {
    auto tOpName = graph[vtxId].toucanOpName;
    auto vtxWeight = graph[vtxId].weight;
    assert(vtxWeight == 1);

    auto op = graph[vtxId].op;
    CGOpMetaInfo opMeta;
    opMeta.opName = tOpName;
    opMeta.op = op;
    opMeta.vtxId = vtxId;
    populateOpMetaDebugInfo(opMeta, op);

    if (tOpName == CGToucanOPName::RegRead) {
      // a regread
      auto regReadOp = cast<toucan::RegReadOp>(op);
      auto regVal = regReadOp.getReg();
      assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register that never seen was read!");
      auto regValId = codeGenInfo.toucanRegToId[regVal];

      opMeta.regRead.reg = regValId;
      currentLevelOps.push_back(opMeta);
    } else if (tOpName == CGToucanOPName::ConstDecl) {
      // A constant decl. Do nothing.
    } else {
      assert(false && "Should not reach here");
    }
  }

  // Allocate result storage
  for (size_t opId = 0; opId < currentLevelOps.size(); opId++) {
    auto &opMeta = currentLevelOps[opId];

    assert(opMeta.opName == CGToucanOPName::RegRead);
    auto valId = partInfo.valuePool.size();

    // new val
    CGValueMetaInfo valMeta;
    valMeta.isConst = false;
    valMeta.isPlaceholder = false;
    valMeta.value = 0;
    valMeta.definingOp = opMeta.op;
    valMeta.levelId = 0;
    // valMeta.opId = opId;
    valMeta.bitWidth = static_cast<uint8_t>(hw::getBitWidth(opMeta.op->getResult(0).getType()));
    valMeta.namehint = opMeta.namehint;
    valMeta.fragment_id = opMeta.fragment_id;

    partInfo.valuePool.push_back(valMeta);
    auto resultVal = opMeta.op->getResult(0);
    assert(!partInfo.valueToValId.contains(resultVal));
    partInfo.valueToValId[resultVal] = valId;
    opMeta.setResult(valId);
  }

  // Save statistics
  CGLayerValueStatistics stats;
  std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
  stats.numRegReads = currentLevelOps.size();
  partInfo.opStatisticsPerLevel.push_back(stats);

  partInfo.opStatistics.numRegReads = currentLevelOps.size();
  // Save ops
  partInfo.opPool.push_back(std::move(currentLevelOps));
}

void MultiRegionScheduler::scheduleMiddleLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &currentLevel, uint32_t levelId) {
  // Codegen for middle layers

  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;

  // lut ops. May be reordered for performance reason
  mlir::SmallVector<CGOpMetaInfo> lutOps;
  // vec produced by nops. ** Don't reorder **
  // each defvector is converted to a list of LUT_Nop
  mlir::SmallVector<mlir::SmallVector<CGOpMetaInfo>> vecDecls;
  mlir::SmallVector<mlir::TypedValue<toucan::VecType>> vecDeclVals;
  // vec reads
  mlir::SmallVector<mlir::SmallVector<CGOpMetaInfo>> vecReadOps;
  mlir::SmallVector<mlir::Value> vecReadHandleVals;
  // mem reads. reorder is not necessary
  mlir::SmallVector<CGOpMetaInfo> memReadOps;

  mlir::SmallVector<CGOpMetaInfo> currentVecReadOps;
  mlir::SmallVector<CGOpMetaInfo> currentVecDeclOps;
  std::array<uint32_t, 3> lutOpValueIds;
  std::array<uint32_t, 4> vecReadOpIndexIds;

  for (auto vtxId: currentLevel) {
    auto tOpName = graph[vtxId].toucanOpName;
    auto rawOp = graph[vtxId].op;
    auto vtxWeight = graph[vtxId].weight;

    CGOpMetaInfo opMeta;
    opMeta.opName = tOpName;
    opMeta.op = rawOp;
    opMeta.vtxId = vtxId;
    populateOpMetaDebugInfo(opMeta, rawOp);

    if (tOpName == CGToucanOPName::LUT) {
      // lut
      assert(vtxWeight == 1);

      auto lutOp = cast<toucan::LUTOp>(rawOp);

      // fill input oprands
      auto lutInputs = lutOp.getInputs();
      auto numLutOprands = lutInputs.size();
      assert(numLutOprands <= 3);
      size_t pos = 0;
      for (; pos < 3 - numLutOprands; pos++) {
        // Note: the first elem in value pool is const 0
        lutOpValueIds[pos] = 0;
      }
      for (auto val: lutOp.getInputs()) {
        assert(partInfo.valueToValId.contains(val));
        auto valId = partInfo.valueToValId[val];
        lutOpValueIds[pos] = valId;
        pos++;
      }

      auto rawOpName = static_cast<uint32_t>(lutOp.getOpName());
      assert(rawOpName <= toucan::getMaxEnumValForLUTOpName());
      opMeta.lut.lutId = static_cast<uint8_t>(rawOpName);
      opMeta.lut.op0 = lutOpValueIds[0];
      opMeta.lut.op1 = lutOpValueIds[1];
      opMeta.lut.op2 = lutOpValueIds[2];
      opMeta.lut.numOprands = numLutOprands;

      populateOpMetaDebugInfo(opMeta, rawOp);

      lutOps.push_back(opMeta);
    } else if (tOpName == CGToucanOPName::VecDecl) {
      // vec decl, expand to list of nops
      currentVecDeclOps.clear();
      auto vecDeclOp = cast<toucan::DefVectorOp>(rawOp);
      // Why reverse? vector decl op elements are MSB first. Reorder to LSB first to make vecRead's life easier
      for (auto elemVal: llvm::reverse(vecDeclOp.getInputs())) {
        // Create a NOP
        CGOpMetaInfo opMeta;
        opMeta.opName = CGToucanOPName::LUT;
        opMeta.op = rawOp;
        opMeta.vtxId = vtxId;
        // nop, the op code should be 0
        opMeta.lut.lutId = static_cast<uint8_t>(LUTOpName::LUT_Nop);
        opMeta.lut.op0 = 0;
        opMeta.lut.op1 = 0;
        assert(partInfo.valueToValId.contains(elemVal));
        opMeta.lut.op2 = partInfo.valueToValId[elemVal];
        opMeta.lut.numOprands = 1;

        // No namehint

        currentVecDeclOps.push_back(opMeta);
      }
      assert(currentVecDeclOps.size() == vtxWeight);

      // Nops are ordered.
      vecDecls.push_back(currentVecDeclOps);
      vecDeclVals.push_back(vecDeclOp.getHandle());
    } else if (tOpName == CGToucanOPName::VecRead) {
      currentVecReadOps.clear();

      auto vecReadOp = cast<toucan::VectorReadOp>(rawOp);
      auto vecHandle = vecReadOp.getHandle();
      assert(partInfo.valueToValId.contains(vecHandle));
      auto vecHandleId = partInfo.valueToValId[vecHandle];
      auto vecLength = vecHandle.getType().getLength();

      for (auto userOp: vecHandle.getUsers()) {
        // Note: Only 1 user has vtxId. Other ops are not included in currentLevelOps
        auto userReadOp = cast<toucan::VectorReadOp>(userOp);

        // offset: i16
        auto offset = userReadOp.getOffset().getZExtValue();
        auto outRangeValue = userReadOp.getOutRangeValue();


        // collect index vecReadOpIndexIds
        auto indexValues = userReadOp.getIndicies();
        auto numIndexValues = indexValues.size();
        assert(numIndexValues <= 4 && "Index is too long");

        size_t pos = 0;
        for (; pos < 4 - numIndexValues; pos++) {
          // Note: the first elem in value pool is const 0
          vecReadOpIndexIds[pos] = 0;
        }
        for (auto val: indexValues) {
          assert(partInfo.valueToValId.contains(val));
          auto valId = partInfo.valueToValId[val];
          vecReadOpIndexIds[pos] = valId;
          pos++;
        }


        CGOpMetaInfo opMeta;
        opMeta.opName = CGToucanOPName::VecRead;
        opMeta.op = userOp;
        // Share same vtxId
        opMeta.vtxId = vtxId;
        // nop, the op code should be 0
        opMeta.vec.vecBase = vecHandleId;
        opMeta.vec.vecLength = vecLength;
        opMeta.vec.index0 = vecReadOpIndexIds[0];
        opMeta.vec.index1 = vecReadOpIndexIds[1];
        opMeta.vec.index2 = vecReadOpIndexIds[2];
        opMeta.vec.index3 = vecReadOpIndexIds[3];

        assert(partInfo.valueToValId.contains(outRangeValue));
        opMeta.vec.outRangeValue = partInfo.valueToValId[outRangeValue];
        opMeta.vec.offset = static_cast<uint16_t>(offset);

        populateOpMetaDebugInfo(opMeta, userOp);

        currentVecReadOps.push_back(opMeta);
      }
      assert(currentVecReadOps.size() == vtxWeight);

      // Sort by offset for performance reason
      std::sort(currentVecReadOps.begin(), currentVecReadOps.end(), 
        [](const CGOpMetaInfo& a, const CGOpMetaInfo& b) {return a.vec.offset < b.vec.offset;});

      vecReadOps.push_back(std::move(currentVecReadOps));
      vecReadHandleVals.push_back(vecHandle);

    } else if (tOpName == CGToucanOPName::MemRead) {
      // a memread
      assert(vtxWeight == 1);

      auto memReadOp = cast<toucan::MemReadOp>(rawOp);
      auto memVal = memReadOp.getMem();
      auto memValId = codeGenInfo.toucanMemToId[memVal];
      auto memEnVal = memReadOp.getEn();
      assert(partInfo.valueToValId.contains(memEnVal));
      auto memEnId = partInfo.valueToValId[memEnVal];

      auto memAddrVec = memReadOp.getAddrVec();
      assert(memAddrVec.getType().getLength() == 8 && "Memory address should be a 32 bit vector");
      auto memAddrVecId = partInfo.valueToValId[memAddrVec];

      // size_t pos = 0;
      // for (; pos < 8 - numMemAddrs; pos++) {
      //   // Note: the first elem in value pool is const 0
      //   memReadOpIndexIds[pos] = 0;
      // }
      // for (auto val: memAddrs) {
      //   assert(partInfo.valueToValId.contains(val));
      //   auto valId = partInfo.valueToValId[val];
      //   memReadOpIndexIds[pos] = valId;
      //   pos++;
      // }

      opMeta.memRead.hasMultipleWriter = codeGenInfo.memPool[memValId].hasMultipleWriter;
      opMeta.memRead.memBase = codeGenInfo.memPool[memValId].memBase;
      opMeta.memRead.memDepth = codeGenInfo.memPool[memValId].memDepth;

      opMeta.memRead.en = memEnId;
      opMeta.memRead.addrVec = memAddrVecId;

      populateOpMetaDebugInfo(opMeta, rawOp);
      
      memReadOps.push_back(opMeta);
    } else {
      assert(false && "other type of op should not appear in middle levels");
    }
  }

  //TODO: need sort for cpu code.

  // Record number of luts/memreads/vecreads for later performance tuning
  CGLayerValueStatistics stats;
  std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
  stats.numMemReads = memReadOps.size();
  stats.numVecReads = vecReadOps.size();
  stats.numLuts = lutOps.size();
  // Note: vecDecl ops are lowered to Nops, which is also lut type
  for (auto &eachVecDeclOps: vecDecls) {
    stats.numLuts += eachVecDeclOps.size();
    partInfo.opStatistics.numLutNops += eachVecDeclOps.size();
  }
  partInfo.opStatisticsPerLevel.push_back(stats);

  partInfo.opStatistics.numMemReads += stats.numMemReads;
  partInfo.opStatistics.numVecReads += stats.numVecReads;
  partInfo.opStatistics.numLuts += stats.numLuts;

  // Within each layer, place memReads first, then vecReads, luts are the last
  // Allocate storage for this layer
  auto expectedOpCount = stats.numMemReads + stats.numVecReads + stats.numLuts;
  currentLevelOps.reserve(expectedOpCount);

  assert(currentLevelOps.size() == 0);
  // save op, allocate result storage
  auto allocateResultForMiddleLayer = [&](CGOpMetaInfo &opMeta) {
    auto valId = partInfo.valuePool.size();
    auto resultVal = opMeta.op->getResult(0);

    CGValueMetaInfo valMeta;
    valMeta.isConst = false;
    valMeta.isPlaceholder = false;
    valMeta.value = 0;
    valMeta.definingOp = opMeta.op;
    valMeta.levelId = levelId;
    // valMeta.opId = currentLevelOps.size();
    valMeta.bitWidth = static_cast<uint8_t>(hw::getBitWidth(opMeta.op->getResult(0).getType()));
    valMeta.namehint = opMeta.namehint;
    valMeta.fragment_id = opMeta.fragment_id;

    assert(!partInfo.valueToValId.contains(resultVal));
    partInfo.valueToValId[resultVal] = valId;
    partInfo.valuePool.push_back(valMeta);

    opMeta.setResult(valId);
    currentLevelOps.push_back(opMeta);
  };

  // place mem reads
  for (auto &opMeta: memReadOps) {
    assert(opMeta.opName == CGToucanOPName::MemRead);
    allocateResultForMiddleLayer(opMeta);
  }

  // place vecReads
  assert(vecReadOps.size() == vecReadHandleVals.size());
  for (auto [singleVecReadOps, vecHandle]: llvm::zip(vecReadOps, vecReadHandleVals)) {
    // each elem inside singleVecReadOps reads vector vecHandle
    for (auto &opMeta: singleVecReadOps) {
      assert(opMeta.opName == CGToucanOPName::VecRead);
      allocateResultForMiddleLayer(opMeta);
    }
  }

  // place luts
  for (auto &opMeta: lutOps) {
    assert(opMeta.opName == CGToucanOPName::LUT);
    allocateResultForMiddleLayer(opMeta);
  }

  // place vecdefs
  for (auto [singleVecDefOps, vecHandle]: llvm::zip(vecDecls, vecDeclVals)) {
    auto bitWidth = static_cast<uint8_t>(vecHandle.getType().getElementWidth());
    for (size_t i = 0; i < singleVecDefOps.size(); ++i) {
      auto &opMeta = singleVecDefOps[i];
      assert(opMeta.opName == CGToucanOPName::LUT);
      assert(opMeta.lut.lutId == static_cast<uint8_t>(toucan::LUTOpName::LUT_Nop));

      auto valId = partInfo.valuePool.size();

      CGValueMetaInfo valMeta;
      valMeta.isConst = false;
      valMeta.value = 0;
      valMeta.definingOp = opMeta.op;
      valMeta.levelId = levelId;
      // valMeta.opId = currentLevelOps.size();
      valMeta.namehint = opMeta.namehint;
      valMeta.bitWidth = bitWidth;
      valMeta.fragment_id = opMeta.fragment_id;
      // First op produces the vecHandle. 
      // Other ops produces an invisible placeholder value
      valMeta.isPlaceholder = (i != 0);
      if (i == 0) {
        assert(!partInfo.valueToValId.contains(vecHandle));
        partInfo.valueToValId[vecHandle] = valId;
      } 
      partInfo.valuePool.push_back(valMeta);

      opMeta.setResult(valId);
      currentLevelOps.push_back(opMeta);
    }
  }

  // Save ops
  partInfo.opPool.push_back(std::move(currentLevelOps));

}

// Note: This step is same as SingleRegionScheduler
void MultiRegionScheduler::scheduleLastLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &lastLevel) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;
  currentLevelOps.reserve(lastLevel.size());

  // Code gen for last level

  mlir::SmallVector<CGOpMetaInfo> regWriteOps;
  mlir::SmallVector<CGOpMetaInfo> memWriteOps;
  mlir::SmallVector<CGOpMetaInfo> printOps;
  mlir::SmallVector<CGOpMetaInfo> stopOps;

  currentLevelOps.clear();
  currentLevelOps.reserve(lastLevel.size());

  for (auto vtxId: lastLevel) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    auto rawOp = graph[vtxId].op;
    auto vtxWeight = graph[vtxId].weight;

    CGOpMetaInfo opMeta;
    opMeta.op = rawOp;
    opMeta.opName = vtxOpName;
    opMeta.vtxId = vtxId;

    if (vtxOpName == CGToucanOPName::RegWrite) {
      // regwrite
      assert(vtxWeight == 1);

      auto regWriteOp = cast<toucan::RegWriteOp>(rawOp);
      auto regVal = regWriteOp.getReg();
      auto dataVal = regWriteOp.getData();

      assert(codeGenInfo.toucanRegToId.contains(regVal) && "A register never seen was written!");
      auto regValId = codeGenInfo.toucanRegToId[regVal];
      assert(partInfo.valueToValId.contains(dataVal) && "A value never seen was read!");
      auto dataValId = partInfo.valueToValId[dataVal];

      opMeta.regWrite.reg = regValId;
      opMeta.regWrite.dat = dataValId;

      // Namehint not needed for last level ops
      regWriteOps.push_back(opMeta);

    } else if (vtxOpName == CGToucanOPName::MemWrite) {
      // memWrite
      // Note: if there is multiple writers, they are all merged into 1 vtx
      // Split
      auto memVal = cast<toucan::MemWriteOp>(rawOp).getMem();
      assert(codeGenInfo.toucanMemToId.contains(memVal));
      auto memValId = codeGenInfo.toucanMemToId[memVal];

      uint32_t numMemWrites = 0;
      for (auto userOp: memVal.getUsers()) {
        if (auto memWriteOp = dyn_cast<toucan::MemWriteOp>(userOp)) {
          // for each writer
          numMemWrites ++;

          CGOpMetaInfo mwOpMeta;
          mwOpMeta.op = rawOp;
          mwOpMeta.opName = vtxOpName;
          mwOpMeta.vtxId = vtxId;

          auto dataVal = memWriteOp.getData();
          auto enVal = memWriteOp.getEn();

          assert(partInfo.valueToValId.contains(dataVal));
          auto dataValId = partInfo.valueToValId[dataVal];
          assert(partInfo.valueToValId.contains(enVal));
          auto enValId = partInfo.valueToValId[enVal];

          auto memAddrVec = memWriteOp.getAddrVec();
          auto memAddrVecId = partInfo.valueToValId[memAddrVec];
          assert(memAddrVec.getType().getLength() == 8 && "Memory address should be 32 bit vector");

          // size_t pos = 0;
          // for (; pos < 8 - numMemAddrs; pos++) {
          //   // Note: the first elem in value pool is const 0
          //   memReadOpIndexIds[pos] = 0;
          // }
          // for (auto val: memAddrs) {
          //   assert(partInfo.valueToValId.contains(val));
          //   auto valId = partInfo.valueToValId[val];
          //   memReadOpIndexIds[pos] = valId;
          //   pos++;
          // }

          mwOpMeta.memWrite.hasMultipleWriter = codeGenInfo.memPool[memValId].hasMultipleWriter;
          mwOpMeta.memWrite.memBase = codeGenInfo.memPool[memValId].memBase;
          mwOpMeta.memWrite.memDepth = codeGenInfo.memPool[memValId].memDepth;

          mwOpMeta.memWrite.addrVec = memAddrVecId;

          mwOpMeta.memWrite.dat = dataValId;
          mwOpMeta.memWrite.en = enValId;

          // Namehint not needed for last level ops
          memWriteOps.push_back(mwOpMeta);
        }
      }
      assert(numMemWrites == vtxWeight);

    } else if (vtxOpName == CGToucanOPName::Print) {
      // print
      assert(vtxWeight == 1);

      auto printOp = cast<toucan::PrintOp>(rawOp);
      auto printStr = printOp.getMsg();
      auto enVal = printOp.getEn();

      assert(partInfo.valueToValId.contains(enVal));
      auto enValId = partInfo.valueToValId[enVal];
      
      assert(codeGenInfo.printStrings.contains(printStr));
      auto printStrId = codeGenInfo.printStrings[printStr];

      opMeta.print.en = enValId;
      opMeta.print.msg = printStrId;

      // Namehint not needed for last level ops
      printOps.push_back(opMeta);

    } else if (vtxOpName == CGToucanOPName::Stop) {
      // stop
      assert(vtxWeight == 1);

      auto stopOp = cast<toucan::StopOp>(rawOp);
      auto enVal = stopOp.getEn();

      assert(partInfo.valueToValId.contains(enVal));
      auto enValId = partInfo.valueToValId[enVal];

      opMeta.stop.en = enValId;

      // Namehint not needed for last level ops
      stopOps.push_back(opMeta);

    } else {
      llvm_unreachable("Other type of ops should not appear in last level");
    }
  }

  // Since none of last level ops produces any output, we don't need allocate storage.
  // Simply push back
  currentLevelOps.append(regWriteOps.begin(), regWriteOps.end());
  currentLevelOps.append(memWriteOps.begin(), memWriteOps.end());
  currentLevelOps.append(printOps.begin(), printOps.end());
  currentLevelOps.append(stopOps.begin(), stopOps.end());

  partInfo.opPool.push_back(std::move(currentLevelOps));

  CGLayerValueStatistics stats;
  std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
  stats.numRegWrites = regWriteOps.size();
  stats.numMemWrites = memWriteOps.size();
  stats.numPrints = printOps.size();
  stats.numStops = stopOps.size();

  partInfo.opStatisticsPerLevel.push_back(stats);
  partInfo.opStatistics.numRegWrites = stats.numRegWrites;
  partInfo.opStatistics.numMemWrites = stats.numMemWrites;
  partInfo.opStatistics.numPrints = stats.numPrints;
  partInfo.opStatistics.numStops = stats.numStops;
    
}

void MultiRegionScheduler::scheduleExchangeReads(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &firstLevel) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;
  currentLevelOps.reserve(firstLevel.size());

  // uint32_t opId = 0;
  for (auto vtxId: firstLevel) {
    auto tOpName = graph[vtxId].toucanOpName;
    auto vtxWeight = graph[vtxId].weight;
    auto exchangeValId = graph[vtxId].exchangeValId;
    auto readVal = codeGenInfo.exchangePool[exchangeValId].val;
    assert(vtxWeight == 1);
    assert(tOpName == CGToucanOPName::ExchangeRead);
    assert(codeGenInfo.exchangePool[exchangeValId].isPadding == false);
    assert(!partInfo.valueToValId.contains(readVal));

    CGOpMetaInfo opMeta;
    opMeta.opName = tOpName;
    opMeta.op = nullptr;
    opMeta.vtxId = vtxId;

    // allocate storage
    auto localValId = partInfo.valuePool.size();
    CGValueMetaInfo valMeta;
    valMeta.isConst = false;
    valMeta.isPlaceholder = false;
    valMeta.value = 0;
    valMeta.definingOp = nullptr;
    valMeta.levelId = 0;
    // valMeta.opId = opId;
    // bitWdith: don't care
    valMeta.bitWidth = 4;
    valMeta.namehint = std::nullopt;
    valMeta.fragment_id = 0;

    partInfo.valuePool.push_back(valMeta);

    partInfo.valueToValId[readVal] = localValId;
    opMeta.setResult(localValId);
    opMeta.exgRead.exchangeVal = exchangeValId;

    currentLevelOps.push_back(opMeta);

    // opId++;
  }

  CGLayerValueStatistics stats;
  std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
  stats.numExchangeReads = currentLevelOps.size();
  partInfo.opStatisticsPerLevel.push_back(stats);

  assert(partInfo.opStatistics.numExchangeReads == 0);
  partInfo.opStatistics.numExchangeReads = currentLevelOps.size();
  // Save ops
  partInfo.opPool.push_back(std::move(currentLevelOps));
}

void MultiRegionScheduler::scheduleExchangeWrites(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &lastLevel) {
  mlir::SmallVector<CGOpMetaInfo> currentLevelOps;
  currentLevelOps.reserve(lastLevel.size());

  for (auto vtxId: lastLevel) {
    auto tOpName = graph[vtxId].toucanOpName;
    auto vtxWeight = graph[vtxId].weight;
    auto exchangeValId = graph[vtxId].exchangeValId;
    auto writeVal = codeGenInfo.exchangePool[exchangeValId].val;
    assert(vtxWeight == 1);
    assert(tOpName == CGToucanOPName::ExchangeWrite);
    assert(codeGenInfo.exchangePool[exchangeValId].isPadding == false);
    assert(partInfo.valueToValId.contains(writeVal));
    auto writeValId = partInfo.valueToValId[writeVal];

    CGOpMetaInfo opMeta;
    opMeta.opName = tOpName;
    opMeta.op = nullptr;
    opMeta.vtxId = vtxId;

    // No need to allocate storage


    opMeta.setResult(exchangeValId);
    opMeta.exgWrite.localVal = writeValId;

    currentLevelOps.push_back(opMeta);
  }

  CGLayerValueStatistics stats;
  std::memset(&stats, 0, sizeof(CGLayerValueStatistics));
  stats.numExchangeWrites = currentLevelOps.size();
  partInfo.opStatisticsPerLevel.push_back(stats);

  assert(partInfo.opStatistics.numExchangeWrites == 0);
  partInfo.opStatistics.numExchangeWrites = currentLevelOps.size();
  // Save ops
  partInfo.opPool.push_back(std::move(currentLevelOps));
}

// Scheduler entry point
void MultiRegionScheduler::schedule(DesignGraph &graph) {
  bool printLevelStat = false;

  auto numRegions = regionGraphs.size();

  // schedule all registers, memories and exchange values
  generateRegMemLayout(graph);

  // dedup strings
  collectPrintString(graph, codeGenInfo.printStrings);

  // llvm::dbgs() << "Verifying original graph\n";

  for (uint32_t regionId = 0; regionId < numRegions; regionId++) {
    if (printLevelStat) {
      llvm::dbgs() << "Region " << regionId << "\n";
    }

    codeGenInfo.regionPartitionIds.emplace_back();

    auto &currentRegionGraph = regionGraphs[regionId];
    auto &currentRegionPartitions = regionPartitions[regionId];
    // auto &currentRegionPartLevels = regionPartLevels[regionId];
    auto numPartitions = currentRegionPartitions.size();

    for (uint32_t partId = 0; partId < numPartitions; partId++) {
      if (printLevelStat) {
        llvm::dbgs() << "Partition " << partId << "\n";
      }

      auto &currentPartLevels = regionPartLevels[regionId][partId];
      auto &firstLevel = currentPartLevels[0];
      auto &lastLevel = currentPartLevels.back();

      auto numLevels = currentPartLevels.size();
      assert(numLevels >= 2);

      CGPartitionMetaInfo partInfo;
      std::memset(&partInfo.opStatistics, 0, sizeof(CGOpStatistics));

      // A const zero for all luts
      CGValueMetaInfo zeroConst = {true, false, 0, nullptr, 0, 0, std::nullopt, 0};
      partInfo.valuePool.push_back(zeroConst);

      if (regionId == 0) {
      // constant only exists in first region. collect them.
        collectConstant(currentRegionGraph, partInfo, firstLevel);
        partInfo.numConstsInValuePool = partInfo.valuePool.size();
        // llvm::dbgs() << "Const pool size " << partInfo.numConstsInValuePool << "\n";
      }


      if (printLevelStat) {
        llvm::dbgs() << "Level 0 has size of " << firstLevel.size() << "\n";
      }
      if (regionId == 0) {
        scheduleFirstLevel(currentRegionGraph, partInfo, codeGenInfo, firstLevel);
      } else {
        scheduleExchangeReads(currentRegionGraph, partInfo, codeGenInfo, firstLevel);
      }

      for (uint32_t levelId = 1; levelId < numLevels - 1; levelId++) {
        // for each middle level
        if (printLevelStat) {
          llvm::dbgs() << "Level " << levelId << " has size of " << currentPartLevels[levelId].size() << "\n";
        }
        auto &currentLevel = currentPartLevels[levelId];
        scheduleMiddleLevel(currentRegionGraph, partInfo, codeGenInfo, currentLevel, levelId);
      }

      if (printLevelStat) {
        llvm::dbgs() << "Last level has size of " << lastLevel.size() << "\n";
      }
      if (regionId < (numRegions - 1)) {
        scheduleExchangeWrites(currentRegionGraph, partInfo, codeGenInfo, lastLevel);
      } else {
        scheduleLastLevel(currentRegionGraph, partInfo, codeGenInfo, lastLevel);
      }
      

      auto partitionFlatId = codeGenInfo.partitionInfo.size();
      codeGenInfo.partitionInfo.push_back(std::move(partInfo));
      codeGenInfo.regionPartitionIds[regionId].push_back(partitionFlatId);
    }
  }
  return;
}
