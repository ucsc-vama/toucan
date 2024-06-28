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


void SchedulerBase::levelizeWorker(const PartitioningGraph &g, mlir::SmallVector<mlir::SmallVector<uint32_t>> &graphLevels) {
  std::vector<uint32_t> topo_order;
  topo_order.reserve(boost::num_vertices(g));
  boost::topological_sort(g, std::back_inserter(topo_order));
  std::reverse(topo_order.begin(), topo_order.end());

  // Initialize levels
  std::vector<uint32_t> levels(boost::num_vertices(g), 0);
  std::vector<uint32_t> sinkVtxes;

  // Assign levels based on dependencies. Ignore sink vtxes
  for (auto v : topo_order) {
    if (boost::out_degree(v, g) == 0) {
      // a sink node
      sinkVtxes.push_back(v);
      levels[v] = UINT32_MAX;
    } else if (boost::in_degree(v, g) != 0) {
      uint32_t max_pred_level = 0;
      for (auto ei = boost::in_edges(v, g); ei.first != ei.second; ++ei.first) {
          auto u = boost::source(*ei.first, g);
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
}

void SchedulerBase::collectPrintString(DesignGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t> &printStrings) {
  uint32_t stringId = 0;

  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    auto vtxOpName = graph.g[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::Print) {
      auto printOp = cast<toucan::PrintOp>(graph.g[vtxId].op);

      auto printStr = printOp.getMsg();
      if (!printStrings.contains(printStr)) {
        // a new string
        printStrings[printStr] = stringId;
        stringId++;
      }
    }
  }
}
