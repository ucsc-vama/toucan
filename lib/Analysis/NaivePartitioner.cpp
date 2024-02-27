#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Pass/AnalysisManager.h"
#include "toucan/ToucanAnalysis.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;

NaivePartitioner::NaivePartitioner(mlir::Operation *op, mlir::AnalysisManager &am) {

  auto graph = am.getAnalysis<DesignGraph>();

  // A simple partitioner that put all vtxes into 1 partition
  partitions.push_back({});
  partitions[0].reserve(boost::num_vertices(graph.g));

  for (auto vp = boost::vertices(graph.g); vp.first != vp.second; ++vp.first) {
    auto vtxId = *vp.first;
    partitions[0].push_back(vtxId);
  }
}


