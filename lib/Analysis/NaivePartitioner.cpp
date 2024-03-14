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

NaivePartitioner::NaivePartitioner(mlir::Operation *op, mlir::AnalysisManager &am) {

  auto graph = am.getAnalysis<DesignGraph>();
  auto numVtxes = boost::num_vertices(graph.g);

  // A simple partitioner that put all vtxes into 1 partition (part 0)
  partitions.push_back({});
  partitions[0].clear();
  partitions[0].resize(numVtxes, 1);
  vtxIdToPartId.resize(numVtxes, 0);

  levelizePartitions(graph);

  schedule(graph);

  fillDebugInfo();

}


