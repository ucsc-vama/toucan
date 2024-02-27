
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"

#include <memory>



#define GEN_PASS_DEF_MERGECONST
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "MergeConstPass"

struct MergeConstPass : toucan::impl::MergeConstBase<MergeConstPass> {
  using MergeConstBase<MergeConstPass>::MergeConstBase;


  void runOnOperation() final {
    // mlir::DenseSet<APInt> constVals;
    // mlir::DenseSet<APInt> constVecs;
    mlir::DenseMap<APInt, Operation*> constVals;
    mlir::DenseMap<APInt, Operation*> constVecs;

    mlir::SmallVector<Operation*> toRemove;

    OpBuilder builder(getOperation());
    IRRewriter rewriter(builder);

    getOperation()->walk([&](toucan::ConstantOp constOp){
      auto constVal = constOp.getValue();
      // llvm::dbgs() << "Const val " << constVal << " width " << constVal.getBitWidth()<< "\n";
      if (constVals.contains(constVal)) {
        // merge
        mergedConsts ++;
        auto replaceOp = constVals[constVal];

        rewriter.replaceOp(constOp, replaceOp);
        toRemove.push_back(constOp);
      } else {
        // first time seen
        constVals[constVal] = constOp;
      }
    });

    getOperation()->walk([&](toucan::DefConstVectorOp constVecOp){
      APInt vecVal;

      auto vecValArray = constVecOp.getValues().getValue();
      for (auto &vecValElem: vecValArray) {
        auto elemVal = cast<mlir::IntegerAttr>(vecValElem).getValue();
        vecVal = vecVal.concat(elemVal);
      }

      if (constVals.contains(vecVal)) {
        // merge
        mergedVecs ++;
        auto replaceOp = constVals[vecVal];
        rewriter.replaceOp(constVecOp, replaceOp);
        toRemove.push_back(constVecOp);
      } else {
        // first time seen
        constVals[vecVal] = constVecOp;
      }
    });

  }
};

std::unique_ptr<mlir::Pass> toucan::createMergeConstPass() {
  return std::make_unique<MergeConstPass>();
}