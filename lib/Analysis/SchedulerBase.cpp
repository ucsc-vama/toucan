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
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <algorithm>


using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


void SchedulerBase::getVtxToLevel(const PartitioningGraph &g, mlir::SmallVector<uint32_t> &levels, uint32_t maxVtxId) {
  std::vector<uint32_t> topo_order;
  mlir::SmallVector<uint32_t> sinkVtxes;

  topo_order.reserve(boost::num_vertices(g));
  boost::topological_sort(g, std::back_inserter(topo_order));
  std::reverse(topo_order.begin(), topo_order.end());

  // Initialize levels
  levels.clear();
  levels.resize(maxVtxId, 0);
  uint32_t maxLevel = 0;

  // Assign levels based on dependencies. Ignore sink vtxes
  for (auto v : topo_order) {
    if (boost::out_degree(v, g) == 0) {
      // a sink node
      sinkVtxes.push_back(v);
    } else if (boost::in_degree(v, g) != 0) {
      uint32_t max_pred_level = 0;
      for (auto ei = boost::in_edges(v, g); ei.first != ei.second; ++ei.first) {
        auto u = boost::source(*ei.first, g);
        max_pred_level = std::max(max_pred_level, levels[u]);
      }
      uint32_t v_level = max_pred_level + 1;
      levels[v] = v_level;
      maxLevel = std::max(maxLevel, v_level);
      assert(v_level < UINT32_MAX);
    }
  }

  assert(!sinkVtxes.empty());

  // move all sinkVtx to last level
  for (auto v: sinkVtxes) {
    levels[v] = maxLevel + 1;
  }
}


void SchedulerBase::levelizeWorker(const PartitioningGraph &g, mlir::SmallVector<mlir::SmallVector<uint32_t>> &graphLevels) {

  mlir::SmallVector<uint32_t> levels;
  getVtxToLevel(g, levels, boost::num_vertices(g));


  for (uint32_t vtx = 0; vtx < levels.size(); vtx++) {
    uint32_t vtxLevel = levels[vtx];
    assert(vtxLevel != UINT32_MAX);
    while (graphLevels.size() <= vtxLevel) {
      graphLevels.emplace_back();
    }
    graphLevels[vtxLevel].push_back(vtx);
  }

  // Every level should have at least 1 nodes
  for (const auto &eachLevel: graphLevels) {
    assert(!eachLevel.empty());
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

void SchedulerBase::populateOpMetaDebugInfo(CGOpMetaInfo &opMeta, mlir::Operation *op) {
  opMeta.namehint = getSVNameHintAttr(op);

  auto fragmentIdAttr = getSignalFragmentIDAttr(op);
  if (fragmentIdAttr) {
    opMeta.fragment_id = fragmentIdAttr->getInt();
  } else {
    opMeta.fragment_id = UINT32_MAX;
  }
}

void SchedulerBase::fillDebugInfo(bool fillSignalDebugInfo) {
  // Consider parallel

  // Collect io signals and extern module signals
  for (size_t regId = 0; regId < codeGenInfo.regPool.size(); regId++) {
    auto &regMeta = codeGenInfo.regPool[regId];
    if (regMeta.isIO) {
      assert(regMeta.namehint && "An io signal must have a name");
      auto namehint = regMeta.namehint.value();
      codeGenInfo.ioSignals.insert(namehint);
    }
  }


  // collect reg info
  for (size_t regId = 0; regId < codeGenInfo.regPool.size(); regId++) {
    auto &regMeta = codeGenInfo.regPool[regId];
    if (regMeta.namehint) {
      // has name hint
      auto namehint = regMeta.namehint.value();
      // every named reg should have a fragment id
      auto fragment_id = regMeta.fragment_id;
      assert(fragment_id != UINT32_MAX);
      if (!codeGenInfo.regDebugInfo.contains(namehint)) {
        codeGenInfo.regDebugInfo.try_emplace(namehint);
      }
      codeGenInfo.regDebugInfo[namehint].push_back(regId);
    }
  }
  // sort by fragment id
  for (auto &elem: codeGenInfo.regDebugInfo) {
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [=](const uint32_t a, const uint32_t b) {
      auto a_fragmentId = codeGenInfo.regPool[a].fragment_id;
      auto b_fragmentId = codeGenInfo.regPool[b].fragment_id;
      // fragment Id should not duplicate
      assert(a_fragmentId != b_fragmentId);
      return a_fragmentId > b_fragmentId;
    });
  }
  // Note: Some register might be removed for optimization purpose, thus the debug info might not be complete.
  /*
  // Verify fragment id correctness
  for (auto &elem: codeGenInfo.regDebugInfo) {
    auto &v = elem.getSecond();
    if (v.size() == 1) continue;
    uint32_t expected_id = v.size() - 1;
    for (const auto &ei: v) {
      assert(expected_id != UINT32_MAX);
      auto fragment_id = codeGenInfo.regPool[ei].fragment_id;
      assert(fragment_id == expected_id);
      expected_id--;
    }
  }
  */



  // collect mem info
  for (size_t memId = 0; memId < codeGenInfo.memPool.size(); memId++) {
    auto &memMeta = codeGenInfo.memPool[memId];
    if (memMeta.namehint) {
      // has name hint
      auto namehint = memMeta.namehint.value();
      // Every named memory should have a fragment id
      auto fragment_id = memMeta.fragment_id;
      assert(fragment_id != UINT32_MAX);
      if (!codeGenInfo.memDebugInfo.contains(namehint)) {
        codeGenInfo.memDebugInfo.try_emplace(namehint);
      }
      codeGenInfo.memDebugInfo[namehint].push_back(memId);
    }
  }
  // sort by fragment id
  for (auto &elem: codeGenInfo.memDebugInfo) {
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [=](const uint32_t a, const uint32_t b) {
      auto a_fragmentId = codeGenInfo.memPool[a].fragment_id;
      auto b_fragmentId = codeGenInfo.memPool[b].fragment_id;
      // fragment Id should not duplicate
      assert(a_fragmentId != b_fragmentId);
      return a_fragmentId > b_fragmentId;
    });
  }
  // Verify fragment id correctness
  for (auto &elem: codeGenInfo.memDebugInfo) {
    auto &v = elem.getSecond();
    if (v.size() == 1) continue;
    uint32_t expected_id = v.size() - 1;
    for (const auto &ei: v) {
      assert(expected_id != UINT32_MAX);
      auto fragment_id = codeGenInfo.memPool[ei].fragment_id;
      assert(fragment_id == expected_id);
      expected_id--;
    }
  }


  if (!fillSignalDebugInfo) {
    return;
  }
  // collect signal info
  for (size_t partId = 0; partId < codeGenInfo.partitionInfo.size(); partId++) {
    auto &partOpPool = codeGenInfo.partitionInfo[partId].opPool;

    for (size_t levelId = 0; levelId < partOpPool.size(); levelId++) {
      auto &currentLevelOps = partOpPool[levelId];
      for (auto &opMeta: currentLevelOps) {
        if (opMeta.hasResult()) {
          if (opMeta.namehint) {
            auto namehint = opMeta.namehint.value();
            if (!codeGenInfo.signalDebugInfo.contains(namehint)) {
              codeGenInfo.signalDebugInfo.try_emplace(namehint);
            }
            auto resultValId = opMeta.getResult();
            codeGenInfo.signalDebugInfo[namehint].push_back({partId, resultValId});
          }
        }
      }
    }
  }

  // Refactor signal namehint
  // Note: some signal might be duplicated. Here we only need 1 copy
  std::unordered_set<uint32_t> signalFragments;
  mlir::SmallVector<std::tuple<uint32_t, uint32_t>> dedupInfos;
  for (auto &elem: codeGenInfo.signalDebugInfo) {
    // auto namehint = elem.getFirst();
    auto &infos = elem.getSecond();

    if (infos.size() > 1) {
      // Possibly duplication
      dedupInfos.clear();
      signalFragments.clear();

      for (const auto &eachFragment: infos) {
        auto partId = std::get<0>(eachFragment);
        auto valId = std::get<1>(eachFragment);
        uint32_t fragment_id = codeGenInfo.partitionInfo[partId].valuePool[valId].fragment_id;

        if (!signalFragments.contains(fragment_id)) {
          // a new fragment
          signalFragments.insert(fragment_id);
          dedupInfos.push_back(eachFragment);
        }
      }

      if (dedupInfos.size() != infos.size()) {
        std::swap(dedupInfos, infos);
      }
    }
  }

  // sort by fragment_id
  for (auto &elem: codeGenInfo.signalDebugInfo) {
    auto &v = elem.getSecond();
    std::sort(v.begin(), v.end(), [=](const std::tuple<uint32_t, uint32_t>& a, const std::tuple<uint32_t, uint32_t> &b) {
      auto a_partId = std::get<0>(a);
      auto b_partId = std::get<0>(b);

      auto a_valId = std::get<1>(a);
      auto b_valId = std::get<1>(b);

      auto a_fragmentId = codeGenInfo.partitionInfo[a_partId].valuePool[a_valId].fragment_id;
      auto b_fragmentId = codeGenInfo.partitionInfo[b_partId].valuePool[b_valId].fragment_id;
      // fragment Id should not duplicate
      assert(a_fragmentId != b_fragmentId);
      return a_fragmentId > b_fragmentId;
    });
  }

  // Don't verify fragment id correctness
  // For regular signals, some fragments can be missing due to optimizations

}


