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


using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


void TwoRegionScheduler::levelizeGraph(DesignGraph &graph) {
  std::vector<uint32_t> topo_order;
  topo_order.reserve(boost::num_vertices(graph.g));
  boost::topological_sort(graph.g, std::back_inserter(topo_order));
  std::reverse(topo_order.begin(), topo_order.end());

  // Initialize levels
  std::vector<uint32_t> levels(boost::num_vertices(graph.g), 0);
  std::vector<uint32_t> sinkVtxes;

  // Assign levels based on dependencies. Ignore sink vtxes
  for (auto v : topo_order) {
    if (boost::out_degree(v, graph.g) == 0) {
      // a sink node
      sinkVtxes.push_back(v);
      levels[v] = UINT32_MAX;
    } else if (boost::in_degree(v, graph.g) != 0) {
      uint32_t max_pred_level = 0;
      for (auto ei = boost::in_edges(v, graph.g); ei.first != ei.second; ++ei.first) {
          auto u = boost::source(*ei.first, graph.g);
          max_pred_level = std::max(max_pred_level, levels[u]);
      }
      uint32_t v_level = max_pred_level + 1;
      levels[v] = v_level;
      assert(v_level < UINT32_MAX);
    }
  }

  assert(!sinkVtxes.empty());


  for (uint32_t vtx = 0; vtx < levels.size(); vtx++) {
    uint32_t vtxLevel = levels[vtx];
    if (vtxLevel == UINT32_MAX) continue;
    while (graphLevels.size() <= vtxLevel) {
      graphLevels.emplace_back();
    }
    graphLevels[vtxLevel].push_back(vtx);
  }

  // Every level should have at least 1 nodes
  for (const auto &eachLevel: graphLevels) {
    assert(!eachLevel.empty());
  }

  // Move all sink vtx to last level
  assert(!sinkVtxes.empty());
  graphLevels.emplace_back();
  for (auto sinkVtx: sinkVtxes) {
    graphLevels.back().push_back(sinkVtx);
  }

  // debug print

  llvm::dbgs() << "Graph has " << graphLevels.size() << " levels\n";
  for (size_t levelId = 0; levelId < graphLevels.size(); levelId++) {
    auto &currentLevel = graphLevels[levelId];
    llvm::dbgs() << "  Level " << levelId << " has " << currentLevel.size() << " verticies\n";
  }
}

uint32_t TwoRegionScheduler::findCutPoint(DesignGraph &graph, float r1Weight) {
  assert(graphLevels.size() > 1);
  assert(r1Weight < 1.0 && "r1Weight sould between 0 to 1");
  assert(r1Weight > 0 && "r1Weight sould between 0 to 1");
  uint32_t graphSize = boost::num_vertices(graph.g);
  uint32_t cutLevel = 0;
  uint32_t r1Vtxes = 0;
  uint32_t r1Target = graphSize * r1Weight;

  for (auto &eachLevel: graphLevels) {
    r1Vtxes += eachLevel.size();
    if (r1Vtxes >= r1Target) {
      return cutLevel;
    }
    cutLevel++;
  }

  llvm_unreachable("Cannot find a cut point. This should not happen");
  return 0;
}

void TwoRegionScheduler::cutGraph(DesignGraph &graph, uint32_t cutLevel) {
  assert(graphLevels.size() > 1);
  assert(cutLevel < graphLevels.size() - 1);
  uint32_t graphSize = boost::num_vertices(graph.g);

  // Map between old vertex to vertex in r1Graph and r2Graph.
  // A node can be in either r1 or r2, thus mixing them together is fine.
  SmallVector<uint32_t> vtxIdToNewId;
  vtxIdToNewId.assign(graphSize, UINT32_MAX);

  SmallVector<uint32_t> r1Vtxes, r2Vtxes;
  BitVector r1VtxSet, r2VtxSet;
  r1VtxSet.resize(graphSize, false);
  r2VtxSet.resize(graphSize, false);
  
  size_t level_id = 0;
  for (; level_id <= cutLevel; level_id++) {
    // Add to region 1
    r1Vtxes.insert(r1Vtxes.end(), graphLevels[level_id].begin(), graphLevels[level_id].end());
  }
  for (; level_id < graphLevels.size(); level_id++) {
    // Add to region 2
    r2Vtxes.insert(r2Vtxes.end(), graphLevels[level_id].begin(), graphLevels[level_id].end());
  }

  // copy nodes for r1Graph and r2Graph
  for (auto &eachVtx: r1Vtxes) {
    auto newVertex = boost::add_vertex(graph.g[eachVtx], r1Graph);
    vtxIdToNewId[eachVtx] = newVertex;
    r1VtxSet.set(eachVtx);
  }
  for (auto &eachVtx: r2Vtxes) {
    auto newVertex = boost::add_vertex(graph.g[eachVtx], r2Graph);
    vtxIdToNewId[eachVtx] = newVertex;
    r2VtxSet.set(eachVtx);
  }

  llvm::dbgs() << "Cut at level " << cutLevel << ", region 1 has " << boost::num_vertices(r1Graph) << " vertices, region 2 has " << boost::num_vertices(r2Graph) << " vertices\n";

  // Add edges
  mlir::DenseMap<uint32_t, uint32_t> writerToExchangeValId;
  uint32_t numExchangeWrite = 0;
  uint32_t numExchangeRead = 0;

  auto rawEdges = boost::edges(graph.g);
  for (auto ei = rawEdges.first; ei != rawEdges.second; ++ei) {
    auto edgeSource = boost::source(*ei, graph.g);
    auto edgeTarget = boost::target(*ei, graph.g);
    auto srcNewId = vtxIdToNewId[edgeSource];
    auto dstNewId = vtxIdToNewId[edgeTarget];

    bool srcInR1 = r1VtxSet[edgeSource];
    bool srcInR2 = r2VtxSet[edgeSource];
    bool dstInR1 = r1VtxSet[edgeTarget];
    bool dstInR2 = r2VtxSet[edgeTarget];

    assert(srcInR1 ^ srcInR2);
    assert(dstInR1 ^ dstInR2);

    if (srcInR1 && dstInR1) {
      // R1 internal edge
      boost::add_edge(srcNewId, dstNewId, r1Graph);
    } else if (srcInR2 && dstInR2) {
      // R2 internal edge
      boost::add_edge(srcNewId, dstNewId, r2Graph);
    } else {
      assert(srcInR1);
      assert(dstInR2);
      // A cross edge, from r1 to r2.

      if (!writerToExchangeValId.contains(edgeSource)) {
        // Allocate a new exchange val, if it's not already exist
        CGExchangeValueMetaInfo valInfo;
        valInfo.writerId = edgeSource;

        uint32_t valId = codeGenInfo.exchangePool.size();
        codeGenInfo.exchangePool.push_back(valInfo);
        writerToExchangeValId[edgeSource] = valId;


        // Create ExchangeWrite for r1
        PartitioningGraphNodeProperty vp;
        vp.op = nullptr;
        vp.weight = 1;
        vp.exchangeValId = valId;
        vp.toucanOpName = CGToucanOPName::ExchangeWrite;

        auto exchangeWriteVtxId = boost::add_vertex(vp, r1Graph);
        boost::add_edge(srcNewId, exchangeWriteVtxId, r1Graph);
        numExchangeWrite++;
      
      }

      auto exchangeValId = writerToExchangeValId[edgeSource];

      // Create ExchangeRead for r2
      PartitioningGraphNodeProperty vp;
      vp.op = nullptr;
      vp.weight = 1;
      vp.exchangeValId = exchangeValId;
      vp.toucanOpName = CGToucanOPName::ExchangeRead;

      auto exchangeReadVtxId = boost::add_vertex(vp, r2Graph);
      boost::add_edge(exchangeReadVtxId, dstNewId, r2Graph);
      // update reader
      codeGenInfo.exchangePool[exchangeValId].readerIds.push_back(edgeTarget);
      numExchangeRead++;
      
    }
  }

  assert(codeGenInfo.exchangePool.size() == writerToExchangeValId.size());
  llvm::dbgs() << "Add " << codeGenInfo.exchangePool.size() << " exchange values, " << numExchangeRead << " ExchangeRead (r2) and " << numExchangeWrite << " ExchangeWrite(r1)\n";
  
  return;
}
