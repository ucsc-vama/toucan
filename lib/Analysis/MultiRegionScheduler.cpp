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


// #define DEBUG_PRINT_LEVEL_STATUS
// #define DEBUG_PRINT_REG_LAYOUT

void MultiRegionScheduler::levelizeGraphForCut(DesignGraph &graph) {
  levelizeWorker(graph.g, graphLevels);

  // debug print
#ifdef DEBUG_PRINT_LEVEL_STATUS
  llvm::dbgs() << "Graph has " << graphLevels.size() << " levels\n";
  for (size_t levelId = 0; levelId < graphLevels.size(); levelId++) {
    auto &currentLevel = graphLevels[levelId];
    llvm::dbgs() << "  Level " << levelId << " has " << currentLevel.size() << " verticies\n";
  }
#endif
}

void MultiRegionScheduler::findCutPoints(DesignGraph &graph) {
  assert(graphLevels.size() > 1);

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


/*
Add NOP lut to break edge directly from input nodes (RegRead, ExchangeRead) to output nodes (RegWrite, ExchangeWrite).
This help the scheduler to create more bulked GPU global memory access
*/
void MultiRegionScheduler::breakDirectIOConnection(DesignGraph &graph) {

  size_t regionId = 0;
  mlir::DenseMap<mlir::Operation*, uint32_t> opToRegionId;
  for (auto & eachRegionGraph: regionGraphs) {
    auto numVtxes = boost::num_vertices(eachRegionGraph);
    for (uint32_t vtxId = 0; vtxId < numVtxes; vtxId++) {
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

    // assume vtxid is continuous. i.e. no vtx removed ever
    auto numVtxes = boost::num_vertices(eachRegionGraph);
    for (uint32_t srcVtx = 0; srcVtx < numVtxes; srcVtx++) {
      auto srcTOpName = eachRegionGraph[srcVtx].toucanOpName;
      bool srcVtxIsExgRead = (srcTOpName == CGToucanOPName::ExchangeRead);

      if (srcTOpName == CGToucanOPName::RegRead || srcTOpName == CGToucanOPName::ExchangeRead) {
        // an input node
        // auto srcOp = eachRegionGraph[srcVtx].op;
        // assert(srcOp == nullptr || isa<toucan::RegReadOp>(srcOp));
        assert(boost::in_degree(srcVtx, eachRegionGraph) == 0);

        mlir::DenseSet<uint32_t> directExchangeContactsToRemove;

        for (auto ei = boost::out_edges(srcVtx, eachRegionGraph); ei.first != ei.second; ei.first++) {
          auto dstVtx = boost::target(*ei.first, eachRegionGraph);
          auto dstTOpName = eachRegionGraph[dstVtx].toucanOpName;
          bool dstVtxIsExgWrite = (dstTOpName == CGToucanOPName::ExchangeWrite);
          // Not for correctness, but direct connection from exgread to exgWrite is unnecessary
          assert(!(srcVtxIsExgRead && dstVtxIsExgWrite) && "Remove this assertion should still works. However a direct edge from ExgRead to ExgWrite is unnecessary.");

          if (dstTOpName == CGToucanOPName::RegWrite || dstTOpName == CGToucanOPName::ExchangeWrite) {
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
    for (uint32_t srcVtx = 0; srcVtx < numVtxes; srcVtx++) {
      auto op = eachRegionGraph[srcVtx].op;
      if (op != nullptr) {
        anyOp = op;
        break;
      }
    }

    auto loc = anyOp->getLoc();
    OpBuilder builder(anyOp);
    IRRewriter rewriter(builder);

    mlir::DenseMap<uint32_t, uint32_t> regToNewReader;
    for (auto [srcVtx, dstVtx]: edgesToBreak) {
      boost::remove_edge(srcVtx, dstVtx, eachRegionGraph);


      if (!regToNewReader.contains(srcVtx)) {
        auto srcOp = eachRegionGraph[srcVtx].op;

        mlir::Value srcVal;

        if (srcOp != nullptr) {
          // a reg read
          assert(eachRegionGraph[srcVtx].toucanOpName == CGToucanOPName::RegRead);
          srcVal = cast<toucan::RegReadOp>(srcOp).getResult();
        } else {
          // a exchange read
          assert(eachRegionGraph[srcVtx].toucanOpName == CGToucanOPName::ExchangeRead);
          auto exchangeValId = eachRegionGraph[srcVtx].exchangeValId;
          assert(codeGenInfo.exchangePool.size() > exchangeValId);
          srcVal = codeGenInfo.exchangePool[exchangeValId].val;
        }


        // create a nop
        auto newNop = rewriter.create<toucan::LUTOp>(loc, toucan::LUTOpName::LUT_Nop, srcVal);


        // Insert nop
        PartitioningGraphNodeProperty vp;
        vp.op = newNop;
        vp.weight = 1;
        vp.exchangeValId = UINT32_MAX;
        vp.toucanOpName = CGToucanOPName::LUT;

        auto nopVtxId = boost::add_vertex(vp, eachRegionGraph);


        srcVal.replaceUsesWithIf(newNop.getResult(), [&](const OpOperand &operand) {
          // tryCount++;
          auto userOp = operand.getOwner();
          auto userRegion = opToRegionId[userOp];

          bool shouldReplace = userRegion > regionId;
          if (vecOpsMovedToLaterRegion.contains(userOp)) {
            auto vecUserRegion = vecOpsMovedToLaterRegion[userOp];
            shouldReplace = vecUserRegion > regionId;
          }

          return shouldReplace;
        });

        // use this nop vtx to avoid direct contact
        boost::add_edge(srcVtx, nopVtxId, eachRegionGraph);
        regToNewReader[srcVtx] = nopVtxId;

        if (eachRegionGraph[srcVtx].toucanOpName == CGToucanOPName::ExchangeRead) {
          auto exchangeValId = eachRegionGraph[srcVtx].exchangeValId;
          codeGenInfo.exchangePool[exchangeValId].readerIds.push_back({regionId, nopVtxId});
        }
      }

      // add edge that use the nop
      auto nopVtxId = regToNewReader[srcVtx];
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
      } else if (dstVtxOpName != CGToucanOPName::RegWrite) {
        assert(false && "Should not reach here");
      }
    }

    llvm::outs() << "Region " << regionId << ": insert " << regToNewReader.size() << " extra NOP verticies to break " << edgesToBreak.size() << " direct IO edges\n";

    regionId++;
  }
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
    assert(regionPartLevels.back()[partId].size() >= 2);
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

  // Here we assume replication rate is relatively small
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regValsWithMultipleReads;
  mlir::DenseMap<mlir::TypedValue<toucan::RegType>, uint32_t> regValToReaderPartId, regValToWriterPartId;
  mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regValRead, regValWrite;

  // Collect reg RW info
  for (size_t partId = 0; partId < partToRegReads.size(); partId++) {
    for (auto vtxId: partToRegReads[partId]) {
      auto regReadOp = cast<toucan::RegReadOp>(graph[vtxId].op);
      auto regVal = regReadOp.getReg();
      regValRead.insert(regVal);

      if (!regValsWithMultipleReads.contains(regVal)) {
        if (regValToReaderPartId.contains(regVal)) {
          // has at least 2 writer
          regValsWithMultipleReads.insert(regVal);
          regValToReaderPartId.erase(regVal);
        } else {
          regValToReaderPartId[regVal] = partId;
        }
      }
    }
  }
  for (size_t partId = 0; partId < partToRegWrites.size(); partId++) {
    for (auto vtxId: partToRegWrites[partId]) {
      auto regWriteOp = cast<toucan::RegWriteOp>(graph[vtxId].op);
      auto regVal = regWriteOp.getReg();
      assert(!regValWrite.contains(regVal) && "Each reg should have only 1 writer");
      regValWrite.insert(regVal);
      regValToWriterPartId[regVal] = partId;
    }
  }


#ifdef DEBUG_PRINT_REG_LAYOUT
  llvm::dbgs() << "In total, there are " << regValsWithMultipleReads.size() << " shared reads\n";
#endif

  // reg vals that read by multiple partitions (multiple reader)
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> groupedSharedReadVals;
  // reg vals that has only 1 reader
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> groupedReadOnceVals;
  // reg vals with no reader
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> groupedWriteOnlyVals;
  // reg vals with no writer
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> sortedReadOnlyVals;

  auto numReadParts = partToRegReads.size();
  auto numWriteParts = partToRegWrites.size();

  groupedSharedReadVals.resize(numWriteParts);
  groupedReadOnceVals.resize(numWriteParts);
  groupedWriteOnlyVals.resize(numWriteParts);


  // Segment by writer, then reader
  // Sort by reader: shared read, read by p0, p1, p2, ...

  // First, group by writer
  for (size_t writerPartId = 0; writerPartId < partToRegWrites.size(); writerPartId++) {
    for (auto vtxId: partToRegWrites[writerPartId]) {
      auto regWriteOp = cast<toucan::RegWriteOp>(graph[vtxId].op);
      auto regVal = regWriteOp.getReg();

      // for each writer section
      if (regValsWithMultipleReads.contains(regVal)) {
        // reg with multiple reader
        groupedSharedReadVals[writerPartId].push_back(regVal);
      } else if (regValToReaderPartId.contains(regVal)) {
        // reg with 1 reader
        groupedReadOnceVals[writerPartId].push_back(regVal);
      } else {
        // write only
        groupedWriteOnlyVals[writerPartId].push_back(regVal);
      }
    }
  
    // Second, within each writer group, further sort by reader

    // TODO: sort sharedVals by what?
    std::sort(groupedReadOnceVals[writerPartId].begin(), groupedReadOnceVals[writerPartId].end(), 
      [&] (const mlir::TypedValue<toucan::RegType>& a, const mlir::TypedValue<toucan::RegType>& b) {
        auto readerPartId_a = regValToReaderPartId[a];
        auto readerPartId_b = regValToReaderPartId[b];
        return readerPartId_a < readerPartId_b;
      });
  }


  // Special handling for read-only vals
  mlir::SmallVector<mlir::TypedValue<toucan::RegType>> readOnlySharedVals, readOnlyOnceVals;

  for (auto &eachReadVal: regValRead) {
    if (!regValToWriterPartId.contains(eachReadVal)) {
      // read only, no writer
      if (regValsWithMultipleReads.contains(eachReadVal)) {
        readOnlySharedVals.push_back(eachReadVal);
      } else {
        assert(regValToReaderPartId.contains(eachReadVal));
        readOnlyOnceVals.push_back(eachReadVal);
      }
    }
  }
  // TODO: sort readOnlySharedVals by what? They are relatively rare
  std::sort(readOnlyOnceVals.begin(), readOnlyOnceVals.end(), 
  [&] (const mlir::TypedValue<toucan::RegType>& a, const mlir::TypedValue<toucan::RegType>& b) {
    auto readerPartId_a = regValToReaderPartId[a];
    auto readerPartId_b = regValToReaderPartId[b];
    return readerPartId_a < readerPartId_b;
  });
  // Merge all together
#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << readOnlySharedVals.size() << " shared read only vals, and " << readOnlyOnceVals.size() << " read only once vals\n";
#endif
  std::copy(readOnlySharedVals.begin(), readOnlySharedVals.end(), std::back_inserter(sortedReadOnlyVals));
  std::copy(readOnlyOnceVals.begin(), readOnlyOnceVals.end(), std::back_inserter(sortedReadOnlyVals));



  // schedule
  for (size_t partId = 0; partId < numWriteParts; partId++) {
#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << "Schedule for writer part " << partId << "\n";
#endif
    regOrdered.emplace_back();

    if (partId == 0) {
      // first group for writer part 0. put read only vals
#ifdef DEBUG_PRINT_REG_LAYOUT
      llvm::dbgs() << "  First writer part. Append all " << sortedReadOnlyVals.size() << " read only regs\n";
#endif
      std::copy(sortedReadOnlyVals.begin(), sortedReadOnlyVals.end(), std::back_inserter(regOrdered.back()));
    }


#ifdef DEBUG_PRINT_REG_LAYOUT
    llvm::dbgs() << "  Append " << groupedSharedReadVals[partId].size() << " share regs\n";
    llvm::dbgs() << "  Append " << groupedReadOnceVals[partId].size() << " read once regs\n";
    llvm::dbgs() << "  Append " << groupedWriteOnlyVals[partId].size() << " write only regs\n";
#endif

    std::copy(groupedSharedReadVals[partId].begin(), groupedSharedReadVals[partId].end(), std::back_inserter(regOrdered.back()));
    std::copy(groupedReadOnceVals[partId].begin(), groupedReadOnceVals[partId].end(), std::back_inserter(regOrdered.back()));
    std::copy(groupedWriteOnlyVals[partId].begin(), groupedWriteOnlyVals[partId].end(), std::back_inserter(regOrdered.back()));
  }

  return;
}

// TODO: Is this the best policy? is it correctly implemented?
void MultiRegionScheduler::groupExchangeVals(mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> &exchangeValIdOrdered) {

  mlir::SmallVector<uint32_t> exchangeValIdToOrder;

  uint32_t exchangeValIdOrderNext = 0;
  exchangeValIdToOrder.resize(codeGenInfo.exchangePool.size(), UINT32_MAX);

  // Temporaries
  mlir::SmallVector<uint32_t> exchangeValShared;
  mlir::SmallVector<mlir::SmallVector<uint32_t>> exchangeValUnique;

  auto numRegions = regionGraphs.size();

  for (size_t regionId = 0; regionId < regionGraphs.size(); regionId++) {
    auto numParts = regionPartLevels[regionId].size();
    const auto &currentRegionGraph = regionGraphs[regionId];
    // llvm::dbgs() <<"Working on region " << regionId << "\n";

    for (size_t partId = 0; partId < numParts; partId++) {

      // Sort exchange write vals
      if ((regionId != (numRegions - 1))) {
        // llvm::dbgs() << "Working on ExchangeWrites\n";
        // Last level ExchangeWrite
        // val used by more than 1 partitons, or cross region
        exchangeValShared.clear();
        // val used by only 1 part
        exchangeValUnique.clear();
        size_t nextRegionNumParts = regionPartLevels[regionId+1].size();
        exchangeValUnique.resize(nextRegionNumParts);

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
              assert(regionNewIdToPartId.size() > regionId+1);
              assert(regionNewIdToPartId[regionId+1].size() > readerVtxId);
              auto userPartId = regionNewIdToPartId[regionId+1][readerVtxId];
              assert(exchangeValUnique.size() > userPartId);
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

void MultiRegionScheduler::sortRegWriteOps(mlir::DenseMap<mlir::Value, uint32_t> &regValToOrder) {

  // regWrite only exists in last level of last region
  auto &currentRegionGraph = regionGraphs.back();
  for (size_t partId = 0; partId < regionPartLevels.back().size(); partId++) {
    auto &lastLevel = regionPartLevels.back()[partId].back();
    mlir::DenseMap<uint32_t, uint32_t> vtxIdToOrder;
    for (const auto &eachVtx: lastLevel) {
      // Default order 0. since we use stable sort they will stay in original order
      uint32_t currentVtxOrder = 0;
      auto tOpName = currentRegionGraph[eachVtx].toucanOpName;
      if (tOpName == CGToucanOPName::RegWrite) {
        auto regWriteOp = cast<toucan::RegWriteOp>(currentRegionGraph[eachVtx].op);
        auto regVal = regWriteOp.getReg();
        assert(regValToOrder.contains(regVal) && "Every register should appear in this map!");
        currentVtxOrder = regValToOrder[regVal];
      }
      assert(!vtxIdToOrder.contains(eachVtx));
      vtxIdToOrder[eachVtx] = currentVtxOrder;
    }
    // sort
    std::stable_sort(lastLevel.begin(), lastLevel.end(), [&](uint32_t a, uint32_t b) {
      assert(vtxIdToOrder.contains(a));
      assert(vtxIdToOrder.contains(b));
      auto a_order = vtxIdToOrder[a];
      auto b_order = vtxIdToOrder[b];
      return a_order < b_order;
    });
  }
}

void MultiRegionScheduler::sortRegReadOps(mlir::DenseMap<mlir::Value, uint32_t> &regValToOrder) {

  // regRead only exists in first level of first region
  auto &currentRegionGraph = regionGraphs[0];
  for (size_t partId = 0; partId < regionPartLevels[0].size(); partId++) {
    auto &firstLevel = regionPartLevels[0][partId][0];
    mlir::DenseMap<uint32_t, uint32_t> vtxIdToOrder;
    for (const auto &eachVtx: firstLevel) {
      // Default order 0. since we use stable sort they will stay in original order
      uint32_t currentVtxOrder = 0;
      auto tOpName = currentRegionGraph[eachVtx].toucanOpName;
      if (tOpName == CGToucanOPName::RegRead) {
        auto regReadOp = cast<toucan::RegReadOp>(currentRegionGraph[eachVtx].op);
        auto regVal = regReadOp.getReg();
        assert(regValToOrder.contains(regVal) && "Every register should appear in this map!");
        currentVtxOrder = regValToOrder[regVal];
      }
      assert(!vtxIdToOrder.contains(eachVtx));
      vtxIdToOrder[eachVtx] = currentVtxOrder;
    }
    // sort
    std::stable_sort(firstLevel.begin(), firstLevel.end(), [&](uint32_t a, uint32_t b) {
      assert(vtxIdToOrder.contains(a));
      assert(vtxIdToOrder.contains(b));
      auto a_order = vtxIdToOrder[a];
      auto b_order = vtxIdToOrder[b];
      return a_order < b_order;
    });
  }
}

void MultiRegionScheduler::sortMiddleLevelOps(uint32_t regionId, uint32_t partId, CGPartitionMetaInfo &partInfo) {
  uint32_t nextOrder = 1;
  mlir::DenseMap<uint32_t, uint32_t> vtxToResultValOrder;
  mlir::DenseMap<uint32_t, uint32_t> vtxToSortOrder;


  auto &currentRegionGraph = regionGraphs[regionId];

  size_t numLevels = regionPartLevels[regionId][partId].size();
  assert(numLevels >= 2);
  for (size_t levelId = 0; levelId < numLevels - 1; levelId++) {
    auto &currentLevelVtxes = regionPartLevels[regionId][partId][levelId];

    // update vtx order
    if (levelId == 0) {
      for (const auto eachVtx: currentLevelVtxes) {
        assert(!vtxToResultValOrder.contains(eachVtx));
        auto tOpName = currentRegionGraph[eachVtx].toucanOpName;

        switch (tOpName) {
          case CGToucanOPName::ConstDecl: {
            vtxToResultValOrder[eachVtx] = 0;
            break;
          }

          case CGToucanOPName::RegRead: {
            auto regReadOp = cast<toucan::RegReadOp>(currentRegionGraph[eachVtx].op);
            auto regVal = regReadOp.getResult();
            assert(partInfo.valueToValId.contains(regVal));
            auto realOrder = partInfo.valueToValId[regVal];
            // May be a pre-allocate location
            vtxToResultValOrder[eachVtx] = realOrder;
            nextOrder = std::max(nextOrder, realOrder);
            // nextOrder = std::max(nextOrder, realOrder % preAllocateStartPos);
            break;
          }

          case CGToucanOPName::ExchangeRead: {
            auto exchangeValId = currentRegionGraph[eachVtx].exchangeValId;
            auto readVal = codeGenInfo.exchangePool[exchangeValId].val;

            auto realOrder = partInfo.valueToValId[readVal];
            // May be a pre-allocate location
            vtxToResultValOrder[eachVtx] = realOrder;
            nextOrder = std::max(nextOrder, realOrder);
            // nextOrder = std::max(nextOrder, realOrder % preAllocateStartPos);
            break;
          }

          default: {
            llvm_unreachable("Op should not appear here");
          }
        }
      }
    } else {
      // Middle levels
      nextOrder++;
      vtxToSortOrder.clear();
      mlir::DenseSet<uint32_t> allInVtxOrder;
      for (auto &eachVtx: currentLevelVtxes) {
        allInVtxOrder.clear();
        auto in_edges_range = boost::in_edges(eachVtx, currentRegionGraph);
        for (auto ei = in_edges_range.first; ei != in_edges_range.second; ++ei) {
          auto srcVtx = boost::source(*ei, currentRegionGraph);
          assert(vtxToResultValOrder.contains(srcVtx));
          auto srcOrder = vtxToResultValOrder[srcVtx];

          allInVtxOrder.insert(srcOrder);
        }
        assert(!allInVtxOrder.empty());
        // Consider: any better policy?
        auto maxOrder = *std::max_element(allInVtxOrder.begin(), allInVtxOrder.end());
        vtxToSortOrder[eachVtx] = maxOrder;
      }

      std::sort(currentLevelVtxes.begin(), currentLevelVtxes.end(), [&](const uint32_t a, const uint32_t b) {
        return vtxToSortOrder[a] < vtxToSortOrder[b];
      });

      // save result order
      for (const auto& eachVtx: currentLevelVtxes) {
        vtxToResultValOrder[eachVtx] = nextOrder;
        nextOrder++;
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


  // 1. sort registers
  // Writer part -> val. Needs padding
  mlir::SmallVector<mlir::SmallVector<mlir::TypedValue<toucan::RegType>>> regPoolOrdered;

  // llvm::dbgs() << "Sorting registers\n";
  sortRegistersForLocality(graph.g, regPoolOrdered);


  // Now, registers and exchangeVal location and ops are sorted
  // Allocate space
  mlir::DenseMap<mlir::Value, uint32_t> regValToOrder;

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

      assert(!regValToOrder.contains(regVal));
      // order 0 reserved for non-reg write ops
      regValToOrder[regVal] = regId + 1;
    }
    // add padding regs
    for (size_t i = 0; i < partitionPaddingSpace; i++) {
      CGRegMetaInfo paddingRegMeta;

      paddingRegMeta.isPadding = true;
      paddingRegMeta.bitWidth = 0;
      paddingRegMeta.fragment_id = UINT32_MAX;
      paddingRegMeta.isIO = false;

      // auto regId = codeGenInfo.regPool.size();
      codeGenInfo.regPool.push_back(paddingRegMeta);
    }
  }

  // Coaleasce memory access.
  sortRegWriteOps(regValToOrder);
  sortRegReadOps(regValToOrder);
  
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
        assert(false && "Every memory should have a fragment id!");
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

  return;
}

// sort exgwrite ops at last level by order of result exchangeVal in exchange pool
void MultiRegionScheduler::sortExchangeWriteOps(const mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> &exchangeValIdOrdered) {
  auto numRegions = regionGraphs.size();
  assert(exchangeValIdOrdered.size() + 1 == numRegions);


  for (uint32_t regionId = 0; regionId < numRegions - 1; regionId++) {
    auto &currentRegionGraph = regionGraphs[regionId];
    for (size_t partId = 0; partId < regionPartLevels[regionId].size(); partId++) {
      auto &lastLevel = regionPartLevels[regionId][partId].back();
      mlir::DenseMap<uint32_t, uint32_t> vtxIdToOrder;
      for (const auto &eachVtx: lastLevel) {
        uint32_t currentVtxOrder = 0;
        auto tOpName = currentRegionGraph[eachVtx].toucanOpName;
        if (tOpName == CGToucanOPName::ExchangeWrite) {
          auto exchangeValId = currentRegionGraph[eachVtx].exchangeValId;
          // Exchange pool is already ordered. Here simply use exchangeValId to order ops
          currentVtxOrder = exchangeValId;
        }
        assert(!vtxIdToOrder.contains(eachVtx));
        vtxIdToOrder[eachVtx] = currentVtxOrder;
      }
      // sort
      std::stable_sort(lastLevel.begin(), lastLevel.end(), [&](uint32_t a, uint32_t b) {
        assert(vtxIdToOrder.contains(a));
        assert(vtxIdToOrder.contains(b));
        auto a_order = vtxIdToOrder[a];
        auto b_order = vtxIdToOrder[b];
        return a_order < b_order;
      });
    }
  }

  return;
}

// sort exgwrite ops at last level by order of result exchangeVal in exchange pool
void MultiRegionScheduler::sortExchangeReadOps(const mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> &exchangeValIdOrdered) {
  auto numRegions = regionGraphs.size();
  assert(exchangeValIdOrdered.size() + 1 == numRegions);


  for (uint32_t regionId = 1; regionId < numRegions; regionId++) {
    auto &currentRegionGraph = regionGraphs[regionId];
    for (size_t partId = 0; partId < regionPartLevels[regionId].size(); partId++) {
      auto &firstLevel = regionPartLevels[regionId][partId][0];
      mlir::DenseMap<uint32_t, uint32_t> vtxIdToOrder;
      for (const auto &eachVtx: firstLevel) {
        uint32_t currentVtxOrder = 0;
        auto tOpName = currentRegionGraph[eachVtx].toucanOpName;
        if (tOpName == CGToucanOPName::ExchangeRead) {
          auto exchangeValId = currentRegionGraph[eachVtx].exchangeValId;
          // Exchange pool is already ordered. Here simply use exchangeValId to order ops
          currentVtxOrder = exchangeValId;
        }
        assert(!vtxIdToOrder.contains(eachVtx));
        vtxIdToOrder[eachVtx] = currentVtxOrder;
      }
      // sort
      std::stable_sort(firstLevel.begin(), firstLevel.end(), [&](uint32_t a, uint32_t b) {
        assert(vtxIdToOrder.contains(a));
        assert(vtxIdToOrder.contains(b));
        auto a_order = vtxIdToOrder[a];
        auto b_order = vtxIdToOrder[b];
        return a_order < b_order;
      });
    }
  }

  return;
}

void MultiRegionScheduler::generateExchangeLayout() {
  // 4. Reorder exchangePool
  // order to exchangeValId
  // region -> writer part -> valId. Needs padding
  mlir::SmallVector<mlir::SmallVector<mlir::SmallVector<uint32_t>>> exchangeValIdOrdered;

  groupExchangeVals(exchangeValIdOrdered);
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
  assert(exchangePoolOrdered.size() == codeGenInfo.exchangePool.size());
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

  sortExchangeWriteOps(exchangeValIdOrdered);
  sortExchangeReadOps(exchangeValIdOrdered);
}

// Collect const decls. DOES NOT collect const vec decls
// Since it's multi-regioned, const vec users might not exists in current region
void MultiRegionScheduler::collectConstantVars(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, const mlir::SmallVector<uint32_t> firstLevelOps) {
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
        // Ignore const vec decls for now.
      }
    }
  }
}

// Collect const vec decls
void MultiRegionScheduler::collectConstantVecs(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, uint32_t partId) {
  // Collect all consts, populate value pool
  // ConstDecl only exists in first level
  auto numVtxes = boost::num_vertices(graph);
  for (uint32_t vtxId = 0; vtxId < numVtxes; vtxId++) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::VecRead) {
      auto op = cast<toucan::VectorReadOp>(graph[vtxId].op);
      auto vecHandle = op.getHandle();
      auto vecDeclOp = vecHandle.getDefiningOp();

      if (auto defConstVecOp = dyn_cast<toucan::DefConstVectorOp>(vecDeclOp)) {
        // a const vector used in this graph/region. 
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

// For performance reason, last level ops with output prefer read from/write to coaleasced mem addrs. For now allocate a temporary id, then replace to correct one later
// by MultiRegionScheduler::mergePreAllocatedLastLevelVals()
void MultiRegionScheduler::preAllocateLastLevel(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &lastLevel) {
  // pre allocate value storage for value needed by MW and EW.

  auto nextPos = preAllocateStartPos;
  for (auto vtxId: lastLevel) {
    auto vtxOpName = graph[vtxId].toucanOpName;
    auto rawOp = graph[vtxId].op;

    if (vtxOpName == CGToucanOPName::RegWrite) {
      // regwrite
      auto regWriteOp = cast<toucan::RegWriteOp>(rawOp);
      auto dataVal = regWriteOp.getData();

      if (!partInfo.valueToValId.contains(dataVal)) {
        // need pre-allocate. Never seen this value
        // partInfo.preAlloc_values.push_back(dataVal);
        partInfo.valueToValId[dataVal] = nextPos;
        // partInfo.valueToValId_preAlloc[dataVal] = nextPos;
        nextPos++;
      }

    } else if (vtxOpName == CGToucanOPName::ExchangeWrite) {
      // exchange write
      auto exchangeValId = graph[vtxId].exchangeValId;
      auto writeVal = codeGenInfo.exchangePool[exchangeValId].val;

      if (!partInfo.valueToValId.contains(writeVal)) {
        // partInfo.preAlloc_values.push_back(writeVal);
        partInfo.valueToValId[writeVal] = nextPos;
        // partInfo.valueToValId_preAlloc[writeVal] = nextPos;
        nextPos++;
      }
    }
  }
  // Reserve space
  uint32_t numPreAllocatedVals = nextPos - preAllocateStartPos;
  assert(numPreAllocatedVals < UINT16_MAX);
  partInfo.preAlloc_valuePool.resize(numPreAllocatedVals);
}

void MultiRegionScheduler::mergePreAllocatedLastLevelVals(CGPartitionMetaInfo &partInfo) {
  uint32_t mergeValuePoolStartPos = partInfo.valuePool.size();
  uint32_t preAllocValueIdStart = preAllocateStartPos;
  uint32_t numOfPreAllocVals = partInfo.preAlloc_valuePool.size();
  assert(numOfPreAllocVals + mergeValuePoolStartPos < UINT16_MAX && "too many values");

  // Move valMeta to the back of value pool
  partInfo.valuePool.insert(partInfo.valuePool.end(), partInfo.preAlloc_valuePool.begin(), partInfo.preAlloc_valuePool.end());
  partInfo.preAlloc_valuePool.clear();

  auto updateValId = [&](uint32_t &valId) {
    if (valId >= preAllocValueIdStart) {
      // need update
      uint32_t newValId = valId - preAllocValueIdStart + mergeValuePoolStartPos;
      assert(newValId < partInfo.valuePool.size());
      valId = newValId;
    }
  };

  // Replace old val with new val
  for (auto &eachLevelOps: partInfo.opPool) {
    for (auto &op: eachLevelOps) {
      auto tOpName = op.opName;

      switch (tOpName) {
        case CGToucanOPName::LUT: {
          updateValId(op.lut.op0);
          updateValId(op.lut.op1);
          updateValId(op.lut.op2);
          updateValId(op.lut.result);
          break;
        }
        case CGToucanOPName::VecRead: {
          assert(op.vec.vecBase < preAllocValueIdStart);
          updateValId(op.vec.index0);
          updateValId(op.vec.index1);
          updateValId(op.vec.index2);
          updateValId(op.vec.index3);
          updateValId(op.vec.outRangeValue);
          updateValId(op.vec.result);
          break;
        }
        case CGToucanOPName::Print: {
          updateValId(op.print.en);
          break;
        }
        case CGToucanOPName::Stop: {
          updateValId(op.stop.en);
          break;
        }
        case CGToucanOPName::RegRead: {
          updateValId(op.regRead.result);
          break;
        }
        case CGToucanOPName::RegWrite: {
          updateValId(op.regWrite.dat);
          break;
        }
        case CGToucanOPName::MemRead: {
          assert(op.memRead.addrVec < preAllocValueIdStart);
          updateValId(op.memRead.en);
          updateValId(op.memRead.result);
          break;
        }
        case CGToucanOPName::MemWrite: {
          assert(op.memWrite.addrVec < preAllocValueIdStart);
          updateValId(op.memWrite.dat);
          updateValId(op.memWrite.en);
          break;
        }
        case CGToucanOPName::ExchangeRead: {
          updateValId(op.exgRead.localVal);
          break;
        }
        case CGToucanOPName::ExchangeWrite: {
          updateValId(op.exgWrite.localVal);
          break;
        }

        // case CGToucanOPName::VecDecl:
        // case CGToucanOPName::ConstDecl:
        //   break;
        default: {
          llvm_unreachable("Unknow op type or should not appear");
        }
      }
    }
  }

}

// Note: This step is same as SingleRegionScheduler
void MultiRegionScheduler::scheduleRegReads(PartitioningGraph &graph, CGPartitionMetaInfo &partInfo, CGInfo &codeGenInfo, const mlir::SmallVector<uint32_t> &firstLevelOps) {
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

    auto resultVal = opMeta.op->getResult(0);

    if (!partInfo.valueToValId.contains(resultVal)) {
      // Result is not pre-allocated
      uint32_t valId = partInfo.valuePool.size();
      assert(valId < preAllocateStartPos && "If you see this error, increase preAllocateStartPos");
      partInfo.valueToValId[resultVal] = valId;
      opMeta.setResult(valId);
      partInfo.valuePool.push_back(valMeta);
    } else {
      // pre-allocated. 
      llvm_unreachable("After insert NOP to break direct edges, RegRead result vals should never be pre-allocated.");
      // uint32_t valId = partInfo.valueToValId[resultVal];
      // assert(valId >= preAllocateStartPos);
      // opMeta.setResult(valId);
      // auto posInPool = valId - preAllocateStartPos;
      // assert(partInfo.preAlloc_valuePool.size() > posInPool);
      // partInfo.preAlloc_valuePool[posInPool] = valMeta;
    }
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
    if (rawOp != nullptr) populateOpMetaDebugInfo(opMeta, rawOp);

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
      assert(vecHandleId < preAllocateStartPos && "Vector value cannot directly write to register or exchange pool, thus should not be pre-allocated!");
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
      assert(memAddrVecId < preAllocateStartPos && "Vector value cannot directly write to register or exchange pool, thus should not be pre-allocated!");
      

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

    if (!partInfo.valueToValId.contains(resultVal)) {
      // Result is not pre-allocated
      uint32_t valId = partInfo.valuePool.size();
      assert(valId < preAllocateStartPos && "If you see this error, increase preAllocateStartPos");
      partInfo.valueToValId[resultVal] = valId;
      opMeta.setResult(valId);
      partInfo.valuePool.push_back(valMeta);
    } else {
      // pre-allocated. 
      uint32_t valId = partInfo.valueToValId[resultVal];
      assert(valId >= preAllocateStartPos);
      opMeta.setResult(valId);
      auto posInPool = valId - preAllocateStartPos;
      assert(partInfo.preAlloc_valuePool.size() > posInPool);
      partInfo.preAlloc_valuePool[posInPool] = valMeta;
    }
    
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
    // ConstDecl may also appear. Do nothing.
    if (tOpName == CGToucanOPName::ConstDecl) continue;
    auto vtxWeight = graph[vtxId].weight;
    auto exchangeValId = graph[vtxId].exchangeValId;
    assert(codeGenInfo.exchangePool.size() > exchangeValId);
    auto readVal = codeGenInfo.exchangePool[exchangeValId].val;
    assert(vtxWeight == 1);
    assert(tOpName == CGToucanOPName::ExchangeRead);
    assert(codeGenInfo.exchangePool[exchangeValId].isPadding == false);

    CGOpMetaInfo opMeta;
    opMeta.opName = tOpName;
    opMeta.op = nullptr;
    opMeta.vtxId = vtxId;

    // allocate storage
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


    opMeta.exgRead.exchangeVal = exchangeValId;

    if (!partInfo.valueToValId.contains(readVal)) {
      // Result is not pre-allocated
      uint32_t valId = partInfo.valuePool.size();
      assert(valId < preAllocateStartPos && "If you see this error, increase preAllocateStartPos");
      partInfo.valueToValId[readVal] = valId;
      opMeta.exgRead.localVal = valId;
      partInfo.valuePool.push_back(valMeta);
    } else {
      // pre-allocated. 
      // TODO: This should be unreachable
      // llvm_unreachable("After insert NOP to break direct edges, ExgRead result vals should never be pre-allocated.");
      uint32_t valId = partInfo.valueToValId[readVal];
      assert(valId >= preAllocateStartPos);
      opMeta.exgRead.localVal = valId;
      auto posInPool = valId - preAllocateStartPos;
      assert(partInfo.preAlloc_valuePool.size() > posInPool);
      partInfo.preAlloc_valuePool[posInPool] = valMeta;
    }

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
    assert(codeGenInfo.exchangePool.size() > exchangeValId);
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

  auto numRegions = regionGraphs.size();

  // schedule all registers, memories and exchange values. Also sort registers and exchange writes
  generateRegMemLayout(graph);
  generateExchangeLayout();

  // dedup strings
  collectPrintString(graph, codeGenInfo.printStrings);
  if (codeGenInfo.printStrings.size() > UINT16_MAX) {
    llvm::errs() << "Too many print strings! (current max is " << UINT16_MAX << ")\n";
    llvm_unreachable("Too many print strings");
  }
  assert(codeGenInfo.printStrings.size() <= UINT16_MAX);
  

  // llvm::dbgs() << "Verifying original graph\n";

  for (uint32_t regionId = 0; regionId < numRegions; regionId++) {
#ifdef DEBUG_PRINT_LEVEL_STATUS
    llvm::dbgs() << "Region " << regionId << "\n";
#endif

    codeGenInfo.regionPartitionIds.emplace_back();

    auto &currentRegionGraph = regionGraphs[regionId];
    auto &currentRegionPartitions = regionPartitions[regionId];
    // auto &currentRegionPartLevels = regionPartLevels[regionId];
    auto numPartitions = currentRegionPartitions.size();

    for (uint32_t partId = 0; partId < numPartitions; partId++) {
#ifdef DEBUG_PRINT_LEVEL_STATUS
      llvm::dbgs() << "Partition " << partId << "\n";
#endif

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
      // constant vars only exists in first region. collect them.
        collectConstantVars(currentRegionGraph, partInfo, firstLevel);
      }
      // const vecs might exists in multiple regions.
      // create const vars for every const vecdecl.
      collectConstantVecs(currentRegionGraph, partInfo, partId);
      partInfo.numConstsInValuePool = partInfo.valuePool.size();
      // llvm::dbgs() << "Region " << regionId << " part " << partId << " has const pool size " << partInfo.numConstsInValuePool << "\n";


#ifdef DEBUG_PRINT_LEVEL_STATUS
      llvm::dbgs() << "Level 0 has size of " << firstLevel.size() << "\n";
#endif

      preAllocateLastLevel(currentRegionGraph, partInfo, codeGenInfo, lastLevel);

      if (regionId == 0) {
        scheduleRegReads(currentRegionGraph, partInfo, codeGenInfo, firstLevel);
      } else {
        scheduleExchangeReads(currentRegionGraph, partInfo, codeGenInfo, firstLevel);
      }

      // After schedule first level, sort middle level ops
      sortMiddleLevelOps(regionId, partId, partInfo);

      for (uint32_t levelId = 1; levelId < numLevels - 1; levelId++) {
        // for each middle level
#ifdef DEBUG_PRINT_LEVEL_STATUS
        llvm::dbgs() << "Level " << levelId << " has size of " << currentPartLevels[levelId].size() << "\n";
#endif
        auto &currentLevel = currentPartLevels[levelId];
        scheduleMiddleLevel(currentRegionGraph, partInfo, codeGenInfo, currentLevel, levelId);
      }

#ifdef DEBUG_PRINT_LEVEL_STATUS
      llvm::dbgs() << "Last level has size of " << lastLevel.size() << "\n";
#endif
      if (regionId < (numRegions - 1)) {
        scheduleExchangeWrites(currentRegionGraph, partInfo, codeGenInfo, lastLevel);
      } else {
        scheduleLastLevel(currentRegionGraph, partInfo, codeGenInfo, lastLevel);
      }

      mergePreAllocatedLastLevelVals(partInfo);

      auto partitionFlatId = codeGenInfo.partitionInfo.size();
      codeGenInfo.partitionInfo.push_back(std::move(partInfo));
      codeGenInfo.regionPartitionIds[regionId].push_back(partitionFlatId);
    }
  }
  return;
}
