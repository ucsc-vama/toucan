
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
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

#include <cstddef>
#include <memory>
#include <string>


#define GEN_PASS_DEF_LOWERCOMBTO4B_REPLICATEOP
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombTo4B_ReplicateOpPass"


struct LowerShortCombReplicateOpTo4B: OpRewritePattern<comb::ReplicateOp> {
  using OpRewritePattern<comb::ReplicateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ReplicateOp repOp, PatternRewriter &rewriter) const final {
    auto inputValue = repOp.getInput();
    if (hw::getBitWidth(inputValue.getType()) != 1) {
      repOp.emitError("ReplicateOp with input width != 1 is not supported");
      return failure();
    }

    auto resultValue = repOp.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    assert(resultValueWidth > 1 && "Why you have a replicateOp with same input and output width?");
    
    if (resultValueWidth > 4) {
      SmallVector<Value> intermediateResults;
      for (auto&& [sigId, sigWidth]: split_signal_4B(resultValueWidth)) {
        if (sigWidth > 1) {
          auto newOp = rewriter.create<toucan::LUTOp>(repOp.getLoc(), LUTOpName::LUT_Rep1b, inputValue);
          auto repVal = newOp.getResult();
          // shink if needed
          auto shrinkVal = removeHighBits(rewriter, repOp.getLoc(), repVal, sigWidth);
          intermediateResults.push_back(shrinkVal);
        } else {
          // 1b. simply use input
          intermediateResults.push_back(inputValue);
        }
      }
      attachNameHintAndFragmentId(rewriter, intermediateResults, getSVNameHintAttr(repOp.getOperation()));

      concat_4b_and_replace(repOp.getOperation(), repOp.getResult(), intermediateResults, rewriter);
    } else {
      auto newOp = rewriter.create<toucan::LUTOp>(repOp.getLoc(), LUTOpName::LUT_Rep1b, inputValue);
      auto repVal = newOp.getResult();
      auto shrinkVal = removeHighBits(rewriter, repOp.getLoc(), repVal, resultValueWidth);

      attachNameHintAndFragmentId(rewriter, shrinkVal.getDefiningOp(), getSVNameHintAttr(newOp));

      rewriter.replaceOp(repOp, shrinkVal);
    }

    return success();
  }
};



struct LowerCombTo4B_ReplicateOpPass : toucan::impl::LowerCombTo4B_ReplicateOpBase<LowerCombTo4B_ReplicateOpPass> {
  using LowerCombTo4B_ReplicateOpBase<LowerCombTo4B_ReplicateOpPass>::LowerCombTo4B_ReplicateOpBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;
  std::shared_ptr<ConversionTarget> target;

  LogicalResult initialize(MLIRContext *context) override {

    RewritePatternSet owningPatterns(context);
    ConversionTarget conversionTarget(*context);
    
    owningPatterns.add<LowerShortCombReplicateOpTo4B>(context);

    conversionTarget.addLegalDialect<toucan::ToucanDialect>();
    conversionTarget.addLegalDialect<hw::HWDialect>();
    conversionTarget.addLegalDialect<comb::CombDialect>();

    // After lowering, following ops should no longer appear
    conversionTarget.addIllegalOp<comb::ReplicateOp>();

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    target = std::make_shared<ConversionTarget>(std::move(
    conversionTarget));

    return success();
  }


  LogicalResult runOnModule(hw::HWModuleOp mod) {
    return applyFullConversion(mod, *target, *patterns);
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<hw::HWModuleOp> modulesToProcess;
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }
    // // Sequential
    // for (auto mod: modulesToProcess) {
    //   auto ret = runOnModule(mod);
    //   if (failed(ret)) return signalPassFailure();
    // }

    // Parallel
    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result)) return signalPassFailure();
  }

};

std::unique_ptr<mlir::Pass> toucan::createLowerCombTo4B_ReplicateOpPass() {
  return std::make_unique<LowerCombTo4B_ReplicateOpPass>();
}
