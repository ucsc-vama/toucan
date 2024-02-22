
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>


#define GEN_PASS_DEF_ENSURETOUCANONLY
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

#include "toucan/ToucanAnalysis.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "EnsureToucanOnlyPass"

struct EnsureToucanOnlyPass : toucan::impl::EnsureToucanOnlyBase<EnsureToucanOnlyPass> {
  using EnsureToucanOnlyBase<EnsureToucanOnlyPass>::EnsureToucanOnlyBase;


  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto analysisResult = getAnalysis<IsLegalToucan4B>();
    
    assert(analysisResult.isLegalToucan4B && "Expect toucan dialect only!");

  }

};

std::unique_ptr<mlir::Pass> toucan::createEnsureToucanOnlyPass() {
  return std::make_unique<EnsureToucanOnlyPass>();
}
