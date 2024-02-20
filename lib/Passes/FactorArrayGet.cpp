
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
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
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <memory>
#include <string>


#define GEN_PASS_DEF_FACTORARRAYGET
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "FactorArrayGetPass"


struct LowerArrayGetWithMux: OpRewritePattern<hw::ArrayGetOp> {
  using OpRewritePattern<hw::ArrayGetOp>::OpRewritePattern;

  static bool arrayIsDirectlyDefined(Operation *arrayDefiningOp) {
    if (isa<hw::ArrayCreateOp>(arrayDefiningOp) || isa<hw::AggregateConstantOp>(arrayDefiningOp)) {
      // Do nothing for direct array access
      return true;
    }
    return false;
  }

  LogicalResult matchAndRewrite(hw::ArrayGetOp op, PatternRewriter &rewriter) const final {
    auto arrayVal = op.getInput();
    auto arrayDefiningOp = arrayVal.getDefiningOp();

    if (arrayIsDirectlyDefined(arrayDefiningOp)) {
      // Do nothing for direct array access
      return failure();
    }

    // else, the array value is a result of op other than hw::array_create and hw::aggreagate_const
    // Currently I only seen mux
    if (auto muxOp = dyn_cast<comb::MuxOp>(arrayDefiningOp)) {
      auto tVal = muxOp.getTrueValue();
      auto fVal = muxOp.getFalseValue();
      auto muxCond = muxOp.getCond();

      if (!(arrayIsDirectlyDefined(tVal.getDefiningOp()) && arrayIsDirectlyDefined(fVal.getDefiningOp()))) {
        return failure();
      }

      auto arrayIndex = op.getIndex();

      auto arrayGetOp_tVal = rewriter.create<hw::ArrayGetOp>(op.getLoc(), tVal, arrayIndex);
      auto arrayGetOp_fVal = rewriter.create<hw::ArrayGetOp>(op.getLoc(), fVal, arrayIndex);
      auto arrayGetVal_tVal = arrayGetOp_tVal.getResult();
      auto arrayGetVal_fVal = arrayGetOp_fVal.getResult();

      auto resultMuxOp = rewriter.create<comb::MuxOp>(op.getLoc(), muxCond, arrayGetVal_tVal, arrayGetVal_fVal);

      rewriter.replaceOp(op, resultMuxOp);
      return success();
    } else {
      op.emitError() << "Unknown operation with hw.array as output type: " << op->getName();
      assert(false);
    }

    return failure();
  }
};




struct FactorArrayGetPass : toucan::impl::FactorArrayGetBase<FactorArrayGetPass> {
  using FactorArrayGetBase<FactorArrayGetPass>::FactorArrayGetBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {

    RewritePatternSet owningPatterns(context);
    
    owningPatterns.add<LowerArrayGetWithMux>(context);

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));

    return success();
  }


  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    return applyPatternsAndFoldGreedily(mod, *patterns);
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<hw::HWModuleOp> modulesToProcess;
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }

    // Parallel
    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result)) return signalPassFailure();
  }

};

std::unique_ptr<mlir::Pass> toucan::createFactorArrayGetPass() {
  return std::make_unique<FactorArrayGetPass>();
}
