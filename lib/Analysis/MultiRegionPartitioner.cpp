#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"

#include "toucan/PartitioningGraph.h"
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
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


#define DEBUG_PRINT_LEVEL_STATUS
// #define DEBUG_PRINT_REG_LAYOUT

void MultiRegionPartitioner::levelizeGraphForCut(DesignGraph &graph) {
  levelizeWorker(graph.g, graphLevels);

  // debug print
#ifdef DEBUG_PRINT_LEVEL_STATUS
  size_t totalVtxes = boost::num_vertices(graph.g);
  size_t accumulatedVtxes = 0;
  llvm::dbgs() << "Graph has " << graphLevels.size() << " levels\n";
  for (size_t levelId = 0; levelId < graphLevels.size(); levelId++) {
    auto &currentLevel = graphLevels[levelId];
    accumulatedVtxes += currentLevel.size();
    float percentage = accumulatedVtxes * 100.0f / totalVtxes;
    llvm::dbgs() << "  Level " << levelId << " has " << currentLevel.size() << " verticies (" << percentage << "%)\n";
  }
#endif
}

void MultiRegionPartitioner::findCutPoints(DesignGraph &graph) {
  assert(graphLevels.size() > 1);

  uint32_t graphSize = boost::num_vertices(graph.g);
  // uint32_t regionTarget = graphSize * cutRatios[0];

  uint32_t currentRegionSize = 0;

  cutPoints.clear();
  // cutPoints.push_back(20);
  return;
  mlir::SmallVector<uint32_t> regionSizes;

  // size_t totalLevels = graphLevels.size();
  // int firstRegionLevels = std::min(static_cast<int>(graphSize * 0.4), 10);
  // cutPoints.push_back(firstRegionLevels);
  int firstRegionLevels = 0, firstRegionSize = 0, currentRegionLevels = 0;
  size_t currentLevel = 0;
  for (auto &eachLevel: graphLevels) {
    currentRegionSize += eachLevel.size();
    currentRegionLevels += 1;
    if (firstRegionLevels != 0) {
      // done with first level
      assert(cutPoints.size() > 0);
      assert(firstRegionSize > 0);
      if (currentRegionLevels > firstRegionLevels) {
        if (currentRegionSize > firstRegionSize * 0.5) {
          // cut
          llvm::outs() << "Cut after level " << currentLevel << ", this region has " << currentRegionSize << " vtxes\n";
          cutPoints.push_back(currentLevel);
          regionSizes.push_back(currentRegionSize);
          currentRegionSize = 0;
          currentRegionLevels = 0;
        }
      }
    }
    if (firstRegionLevels == 0 && (static_cast<size_t>(graphSize * 0.4) <= currentRegionSize || currentLevel >= 12)) {
      // save as first level
      llvm::outs() << "Cut after level " << currentLevel << ", this region has " << currentRegionSize << " vtxes\n";
      firstRegionLevels = currentLevel;
      cutPoints.push_back(firstRegionLevels);
      firstRegionSize = currentRegionSize;
      regionSizes.push_back(currentRegionSize);
      currentRegionSize = 0;
      currentRegionLevels = 0;
    }
    currentLevel++;
  }
  regionSizes.push_back(currentRegionSize);

  // Merge last region to previous one if it's too small
  if (regionSizes.size() > 2) {
    auto numRegions = regionSizes.size();
    if (regionSizes.back() < (regionSizes[numRegions-2] * 0.4)) {
      cutPoints.pop_back();
    }
  }


  return;
}

// No need to cut.
void MultiRegionPartitioner::doNotCutGraph(DesignGraph &graph) {
  assert(cutPoints.empty());
  regionGraphs.clear();
  regionGraphs.emplace_back();

  regionGraphs[0].copy_from(graph.g);

  uint32_t numVtxes = boost::num_vertices(graph.g);

  regionNewIdToVtxId.clear();
  regionNewIdToVtxId.emplace_back();
  regionNewIdToVtxId[0].reserve(numVtxes);
  for (uint32_t vtxId = 0; vtxId < numVtxes; vtxId++) {
    regionNewIdToVtxId[0].push_back(vtxId);
  }
}

void MultiRegionPartitioner::cutGraph(DesignGraph &graph) {
  if (cutPoints.empty()) {
    doNotCutGraph(graph);
    return;
  }
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

  // Move VecDecl and const vec decl to its user region (backwards)
  mlir::SmallVector<uint32_t> vecUserRegions;
  for (uint32_t vtxId = 0; vtxId < graphSize; vtxId++) {
    auto tOpName = graph.g[vtxId].toucanOpName;
    auto opPtr = graph.g[vtxId].op;
    assert(opPtr != nullptr);
    assert(!opToVtxId.contains(opPtr));
    opToVtxId[opPtr] = vtxId;

    bool is_VecDecl_or_ConstVecDecl = false;

    if (tOpName == CGToucanOPName::VecDecl) {
      // A vec decl
      is_VecDecl_or_ConstVecDecl = true;
    } else if (tOpName == CGToucanOPName::ConstDecl && isa<toucan::DefConstVectorOp>(opPtr)) {
      // A const vec decl
      is_VecDecl_or_ConstVecDecl = true;
    }


    if (is_VecDecl_or_ConstVecDecl) {
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
        vecDeclMovedToLaterRegion[opPtr] = vecUserRegion;
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

    auto srcOpName = graph.g[edgeSource].toucanOpName;

    if (srcOpName == CGToucanOPName::ConstDecl && srcRegion != dstRegion) {
      // Consts are pinned at the beginning of every region's value pool.
      // Remove cross region edges if the source is a const
      continue;
    }

    if (srcRegion == dstRegion) {
      // Internal edge
      boost::add_edge(srcNewId, dstNewId, regionGraphs[srcRegion]);
    } else {
      // An edge that cross (possibly multipe) regions
      assert(srcRegion < dstRegion);

      auto srcOpName = graph.g[edgeSource].toucanOpName;
      auto srcOpCount = graph.g[edgeSource].opCount;

      if (srcOpCount == 1) {
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
          vp.weight = DESIGNGRAPH_EXGWRITE_WEIGHT;
          vp.opCount = 0;
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
          vp.weight = DESIGNGRAPH_EXGREAD_WEIGHT;
          vp.opCount = 0;
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
        // Even though it's possible for a VecDecl to have op count more than 1,
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

          assert(llvm::range_size(vecUsers) == srcOpCount);

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
            vp.weight = DESIGNGRAPH_EXGWRITE_WEIGHT;
            vp.opCount = 0;
            vp.exchangeValId = valId;
            vp.toucanOpName = CGToucanOPName::ExchangeWrite;

            auto exchangeWriteVtxId = boost::add_vertex(vp, regionGraphs[srcRegion]);
            // TODO: bug fix
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
                  vp.weight = DESIGNGRAPH_EXGREAD_WEIGHT;
                  vp.opCount = 0;
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

                // TODO: this assertion is not correct
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


/*
Add NOP lut to break edge directly from input nodes (RegRead, ExchangeRead) to output nodes (RegWrite, ExchangeWrite).
This help the scheduler to create more bulked GPU global memory access

Note: Also break StaticVectorSegmentReadOp to RW
*/
void MultiRegionPartitioner::breakDirectIOConnection() {

  size_t regionId = 0;
  mlir::DenseMap<mlir::Operation*, uint32_t> opToRegionId;
  for (auto & eachRegionGraph: regionGraphs) {
    for (auto vtxId : boost::make_iterator_range(vertices(eachRegionGraph))) {
      auto op = eachRegionGraph[vtxId].op;
      if (op != nullptr) {
        assert(!opToRegionId.contains(op));
        opToRegionId[op] = regionId;
      }
    }
    regionId++;
  }

  // mlir::DenseSet<mlir::Operation*> vecOpsWithValReplaced;
  mlir::DenseMap<mlir::Operation*, uint32_t> vecOpsMovedToLaterRegion;
  for (auto [vecDecl, vecUserRegion]: vecDeclMovedToLaterRegion) {
    vecOpsMovedToLaterRegion[vecDecl] = vecUserRegion;
    for (auto eachVecUser: vecDecl->getUsers()) {
      vecOpsMovedToLaterRegion[eachVecUser] = vecUserRegion;
    }
  }


  regionId = 0;
  for (auto & eachRegionGraph: regionGraphs) {
    mlir::SmallVector<std::pair<uint32_t, uint32_t>> edgesToBreak;

    for (auto srcVtx : boost::make_iterator_range(vertices(eachRegionGraph))) {
      auto srcTOpName = eachRegionGraph[srcVtx].toucanOpName;
      bool srcVtxIsExgRead = (srcTOpName == CGToucanOPName::ExchangeRead);
      bool srcVtxIsRegRead = (srcTOpName == CGToucanOPName::RegRead);
      bool srcVtxIsVecArith = (srcTOpName == CGToucanOPName::VecArith);

      if (srcVtxIsRegRead || srcVtxIsExgRead || srcVtxIsVecArith) {
        // an input node
        // auto srcOp = eachRegionGraph[srcVtx].op;
        // assert(srcOp == nullptr || isa<toucan::RegReadOp>(srcOp));

        // Check graph node input degree
        if (!srcVtxIsVecArith) {
          assert(boost::in_degree(srcVtx, eachRegionGraph) == 0);
        } else {
          // vec arith, it's possible to have 2 const vec as input
          auto vecArithOp = cast<toucan::VectorArithOp>(eachRegionGraph[srcVtx].op);
          auto v1 = vecArithOp.getV1();
          auto v2 = vecArithOp.getV2();
          auto v1IsConstVec = isa<toucan::DefConstVectorOp>(v1.getDefiningOp());
          auto v2IsConstVec = isa<toucan::DefConstVectorOp>(v2.getDefiningOp());

          auto isConstVecArith = v1IsConstVec && v2IsConstVec;

          if (!isConstVecArith) {
            assert(boost::in_degree(srcVtx, eachRegionGraph) != 0);
          } else {
            assert(boost::in_degree(srcVtx, eachRegionGraph) == 0);
          }
        }

        mlir::DenseSet<uint32_t> directExchangeContactsToRemove;

        for (auto ei = boost::out_edges(srcVtx, eachRegionGraph); ei.first != ei.second; ei.first++) {
          auto dstVtx = boost::target(*ei.first, eachRegionGraph);
          auto dstTOpName = eachRegionGraph[dstVtx].toucanOpName;
          bool dstVtxIsExgWrite = (dstTOpName == CGToucanOPName::ExchangeWrite);
          // Not for correctness, but direct connection from exgread to exgWrite is unnecessary
          assert(!(srcVtxIsExgRead && dstVtxIsExgWrite) && "Remove this assertion should still works. However a direct edge from ExgRead to ExgWrite is unnecessary.");

          if (dstTOpName == CGToucanOPName::RegWrite 
            || dstTOpName == CGToucanOPName::ExchangeWrite
            || dstTOpName == CGToucanOPName::Stop
            || dstTOpName == CGToucanOPName::Print) {
            // Need to break such edge
            assert(boost::out_degree(dstVtx, eachRegionGraph) == 0);
            edgesToBreak.push_back({srcVtx, dstVtx});
            directExchangeContactsToRemove.insert(dstVtx);
          }
        }

        // update exchange meta info
        if (srcVtxIsExgRead && !directExchangeContactsToRemove.empty()) {
          mlir::SmallVector<std::tuple<uint32_t, uint32_t>> newReaderIds;
          auto exchangeValId = eachRegionGraph[srcVtx].exchangeValId;
          for (auto [rRegion, rVtx]: codeGenInfo.exchangePool[exchangeValId].readerIds) {
            if (!(rRegion == regionId && directExchangeContactsToRemove.contains(rVtx))){
              newReaderIds.push_back({regionId, rVtx});
            }
          }

          // TODO: Found some issue here: readerIds may not be correct

          // assert(newReaderIds.size() < codeGenInfo.exchangePool[exchangeValId].readerIds.size());
          std::swap(codeGenInfo.exchangePool[exchangeValId].readerIds, newReaderIds);
        }

      }
    }

    // pick any operation to create IRRewriter
    mlir::Operation *anyOp = nullptr;
    for (auto srcVtx : boost::make_iterator_range(vertices(eachRegionGraph))) {
      auto op = eachRegionGraph[srcVtx].op;
      if (op != nullptr) {
        anyOp = op;
        break;
      }
    }

    auto loc = anyOp->getLoc();
    OpBuilder builder(anyOp);
    IRRewriter rewriter(builder);

    // mlir::DenseMap<uint32_t, uint32_t> regToNewReader;
    mlir::DenseMap<mlir::Value, uint32_t> valToNewReader;
    for (auto [srcVtx, dstVtx]: edgesToBreak) {
      auto dstTOpName = eachRegionGraph[dstVtx].toucanOpName;
      bool dstVtxIsExgWrite = (dstTOpName == CGToucanOPName::ExchangeRead);
      bool dstVtxIsRegWrite = (dstTOpName == CGToucanOPName::RegWrite);
      bool dstVtxIsStop = (dstTOpName == CGToucanOPName::Stop);
      bool dstVtxIsPrint = (dstTOpName == CGToucanOPName::Print);

      boost::remove_edge(srcVtx, dstVtx, eachRegionGraph);

      mlir::Value srcVal;
      if (dstVtxIsRegWrite) {
        auto dstOp = eachRegionGraph[dstVtx].op;
        assert(isa<toucan::RegWriteOp>(dstOp));
        srcVal = cast<toucan::RegWriteOp>(dstOp).getData();
      } else if (dstVtxIsPrint) {
        srcVal = cast<toucan::PrintOp>(eachRegionGraph[dstVtx].op).getEn();
      } else if (dstVtxIsStop) {
        srcVal = cast<toucan::StopOp>(eachRegionGraph[dstVtx].op).getEn();
      } else if (dstVtxIsExgWrite) {
        assert(false && "This procedure is not tested!!!");
        assert(dstVtxIsExgWrite);
        auto exchangeValId = eachRegionGraph[dstVtx].exchangeValId;
        assert(codeGenInfo.exchangePool.size() > exchangeValId);
        srcVal = codeGenInfo.exchangePool[exchangeValId].val;
      } else {
        dbgs() << stringifyCGToucanOPName(dstTOpName) << "\n";
        llvm_unreachable("Unexpected break edge end point");
      }


      if (!valToNewReader.contains(srcVal)) {
        // create a nop
        auto newNop = rewriter.create<toucan::LUTOp>(loc, toucan::LUTOpName::LUT_Nop, srcVal);


        // Insert nop
        PartitioningGraphNodeProperty vp;
        vp.op = newNop;
        vp.weight = DESIGNGRAPH_BREAK_IO_NOP_WEIGHT;
        vp.opCount = 1;
        vp.exchangeValId = UINT32_MAX;
        vp.toucanOpName = CGToucanOPName::LUT;

        auto nopVtxId = boost::add_vertex(vp, eachRegionGraph);


        srcVal.replaceUsesWithIf(newNop.getResult(), [&](const OpOperand &operand) {
          auto userOp = operand.getOwner();

          if (!opToRegionId.contains(userOp)) {
            // Could be new NOP op. don't replace it.
            // Or could be merge ops. Don't replace.
            if (!(isa<toucan::DefVectorOp>(userOp)
               || isa<toucan::VectorReadOp>(userOp))) {
              // must be the NOP
              assert(userOp == newNop);
            }
            return false;
          }

          auto userRegion = opToRegionId[userOp];

          // some vector's user region may be moved to later regions.
          if (vecOpsMovedToLaterRegion.contains(userOp)) {
            userRegion = vecOpsMovedToLaterRegion[userOp];
          }

          bool shouldReplace = userRegion > regionId;

          // For use within current region, replace if it's a terminal vtx
          if (!shouldReplace && (
            isa<toucan::RegWriteOp>(userOp)
            || isa<toucan::PrintOp>(userOp)
            || isa<toucan::StopOp>(userOp)
          )) {
            shouldReplace = true;
          }

          return shouldReplace;
        });

        // use this nop vtx to avoid direct contact
        boost::add_edge(srcVtx, nopVtxId, eachRegionGraph);
        assert(!valToNewReader.contains(srcVal));
        valToNewReader[srcVal] = nopVtxId;

        if (eachRegionGraph[srcVtx].toucanOpName == CGToucanOPName::ExchangeRead) {
          auto exchangeValId = eachRegionGraph[srcVtx].exchangeValId;
          codeGenInfo.exchangePool[exchangeValId].readerIds.push_back({regionId, nopVtxId});
        }
      }

      // add edge that use the nop
      auto nopVtxId = valToNewReader[srcVal];
      boost::add_edge(nopVtxId, dstVtx, eachRegionGraph);

      // update use
      auto nopVal = cast<toucan::LUTOp>(eachRegionGraph[nopVtxId].op).getResult();
      auto dstVtxOpName = eachRegionGraph[dstVtx].toucanOpName;
      auto dstOp = eachRegionGraph[dstVtx].op;

      if (dstVtxOpName == CGToucanOPName::ExchangeWrite) {
        // update exchangeVal
        assert(dstOp == nullptr);
        auto exchangeValId = eachRegionGraph[dstVtx].exchangeValId;
        assert(codeGenInfo.exchangePool.size() > exchangeValId);
        codeGenInfo.exchangePool[exchangeValId].val = nopVal;
        codeGenInfo.exchangePool[exchangeValId].writerId = nopVtxId;
      }
    }

    llvm::outs() << "Region " << regionId << ": insert " << valToNewReader.size() << " extra NOP verticies to break " << edgesToBreak.size() << " direct IO edges\n";

    regionId++;
  }
}

LogicalResult MultiRegionPartitioner::levelizeAllPartitions(mlir::MLIRContext *context) {
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
      // has ExchangeRead, then the first level must be exchange reads or const Decl
      for (auto vtx: regionLevels[0]) {
        assert(regionGraphs[regionId][vtx].toucanOpName == CGToucanOPName::ExchangeRead || regionGraphs[regionId][vtx].toucanOpName == CGToucanOPName::ConstDecl);
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

  

#ifdef DEBUG_PRINT_LEVEL_STATUS
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
#endif

  return success();
}

