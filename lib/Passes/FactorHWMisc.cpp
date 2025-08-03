
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
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
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <memory>


#define GEN_PASS_DEF_FACTORHWMISC
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "FactorHWMiscPass"



std::atomic<uint64_t> wireOpRemovedInModule;
std::atomic<uint64_t> hierpathOpRemovedInModule;

struct LowerHWWire: OpRewritePattern<hw::WireOp> {
  using OpRewritePattern<hw::WireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hw::WireOp op, PatternRewriter &rewriter) const final {

    auto inputVal = op.getInput();
    rewriter.replaceOp(op, inputVal);

    auto name = op.getName();
    if (name) {
      auto nameAttr = rewriter.getStringAttr(name.value());
      setSVNameHintAttr(op, nameAttr);
    }

    wireOpRemovedInModule++;

    return success();
  }
};

struct LowerHWHierPath: OpRewritePattern<hw::HierPathOp> {
  using OpRewritePattern<hw::HierPathOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hw::HierPathOp op, PatternRewriter &rewriter) const final {
    op.erase();
    wireOpRemovedInModule++;
    return success();
  }
};


struct FactorHWMiscPass : toucan::impl::FactorHWMiscBase<FactorHWMiscPass> {
  using FactorHWMiscBase<FactorHWMiscPass>::FactorHWMiscBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;
  std::shared_ptr<ConversionTarget> target;

  LogicalResult initialize(MLIRContext *context) override {
    wireOpRemovedInModule = 0;
    hierpathOpRemovedInModule = 0;

    RewritePatternSet owningPatterns(context);
    ConversionTarget conversionTarget(*context);
    
    owningPatterns.add<LowerHWWire>(context);
    owningPatterns.add<LowerHWHierPath>(context);

    conversionTarget.addLegalDialect<toucan::ToucanDialect>();
    conversionTarget.addLegalDialect<hw::HWDialect>();
    conversionTarget.addLegalDialect<comb::CombDialect>();
    conversionTarget.addLegalDialect<seq::SeqDialect>();

    // After lowering, following ops should no longer appear
    conversionTarget.addIllegalOp<hw::WireOp>();


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


    wireOpRemoved = wireOpRemovedInModule;
    hierpathOpRemoved = hierpathOpRemovedInModule;
  }

};

std::unique_ptr<mlir::Pass> toucan::createFactorHWMiscPass() {
  return std::make_unique<FactorHWMiscPass>();
}
