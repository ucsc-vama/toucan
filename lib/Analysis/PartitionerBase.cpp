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
#include <cstdint>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

void PartitionerBase::levelizePartitions(DesignGraph &graph) {
  for (auto &part: partitions) {
    // for each partition
    mlir::SmallVector<mlir::SmallVector<uint64_t>> currentPart;
    mlir::SmallVector<uint64_t> currentLevel;

    // TODO: implement using topological sort

    // // first, collect all source nodes;
    // for (auto vp = boost::vertices(graph.g); vp.first != vp.second; ++vp.first) {
    //   auto vtxId = *vp.first;
    //   if (boost::in_degree(vtxId, graph.g) == 0) {
    //     // No input. put into level 0
    //     currentLevel.push_back(vtxId);
    //   }
    // }
  }
  return;
}


