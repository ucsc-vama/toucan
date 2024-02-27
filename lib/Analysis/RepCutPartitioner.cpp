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

RepCutPartitioner::RepCutPartitioner(mlir::Operation *op, mlir::AnalysisManager &am) {
  // TODO
  assert(false && "Not implemented");
}


