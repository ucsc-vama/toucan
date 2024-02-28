#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <_types/_uint64_t.h>
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <iterator>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

void PartitionerBase::levelizePartitions(DesignGraph &graph) {
  std::vector<uint64_t> topo_order;
  topo_order.reserve(boost::num_vertices(graph.g));
  boost::topological_sort(graph.g, std::back_inserter(topo_order));
  std::reverse(topo_order.begin(), topo_order.end());

  // Initialize levels
  std::vector<uint64_t> levels(boost::num_vertices(graph.g), 0);
  
  // Assign levels based on dependencies
  for (auto v : topo_order) {
    if (boost::in_degree(v, graph.g) != 0) {
      uint64_t max_pred_level = 0;
      for (auto ei = boost::in_edges(v, graph.g); ei.first != ei.second; ++ei.first) {
          auto u = boost::source(*ei.first, graph.g);
          max_pred_level = std::max(max_pred_level, levels[u]);
      }
      levels[v] = max_pred_level + 1;
    }
  }

  // init
  for (size_t partId = 0; partId < partitions.size(); ++partId) {
    partLevels.push_back({});
    partLevels[partId].push_back({});
  }

  // save level in partition
  for (uint64_t vtx = 0; vtx < boost::num_vertices(graph.g); ++vtx) {
    auto vtxLevel = levels[vtx];
    auto vtxPartition = vtxIdToPartId[vtx];
    // grow container if needed
    for (size_t levelId = partLevels[vtxPartition].size(); levelId <= vtxLevel; levelId++) {
      partLevels[vtxPartition].push_back({});
    }
    partLevels[vtxPartition][vtxLevel].push_back(vtx);
  }

  // Condense, remove empty levels
  partLevels.erase(std::remove_if(partLevels.begin(), partLevels.end(), [](auto innerVec) {
    return innerVec.empty();
  }), partLevels.end());
  
  // debug
  for (size_t partId = 0; partId < partLevels.size(); partId++) {
    auto &currentPart = partLevels[partId];
    llvm::dbgs() << "Partition " << partId << " has " << currentPart.size() << " levels\n";
    for (size_t levelId = 0; levelId < currentPart.size(); levelId++) {
      auto &currentLevel = currentPart[levelId];
      llvm::dbgs() << "  Level " << levelId << " has " << currentLevel.size() << " verticies\n";

      // if (currentLevel.size() == 1) {
      //   // level with only 1 node. 
      //   auto vtxId = currentLevel.front();
      //   auto op = graph.g[vtxId].op;
      //   op->print(llvm::dbgs());
      // }
    }
  }
  return;
}


