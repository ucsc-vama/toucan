
#include "circt/Dialect/HW/HWDialect.h"
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
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>


#define GEN_PASS_DEF_TOUCANCANONICALIZER
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "ToucanCanonicalizerPass"




struct ToucanCanonicalizerPass : toucan::impl::ToucanCanonicalizerBase<ToucanCanonicalizerPass> {
  using ToucanCanonicalizerBase<ToucanCanonicalizerPass>::ToucanCanonicalizerBase;

  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;

  ArrayRef<std::string> disabledPatterns;
  ArrayRef<std::string> enabledPatterns;

  ToucanCanonicalizerPass() = default;
  ToucanCanonicalizerPass(const GreedyRewriteConfig &config)
      : config(config) { }

  ToucanCanonicalizerPass(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  LogicalResult initialize(MLIRContext *context) override {

    RewritePatternSet owningPatterns(context);
    // Don't run ToucanCanonicalizer for hw dialect!
    for (auto *dialect : context->getLoadedDialects()) {
      if (isa<toucan::ToucanDialect>(dialect)) {
        dialect->getCanonicalizationPatterns(owningPatterns);
      }
    }
      
    for (RegisteredOperationName op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(owningPatterns, context);
    }

    patterns = std::make_shared<FrozenRewritePatternSet>(
        std::move(owningPatterns), disabledPatterns, enabledPatterns);
    return success();
  }


  // void runOnOperation() override {
  //    LogicalResult converged =
  //        applyPatternsAndFoldGreedily(getOperation(), *patterns, config);
  //    // Canonicalization is best-effort. Non-convergence is not a pass failure.
  //    if (testConvergence && failed(converged))
  //      signalPassFailure();
  //  }

  

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    auto converged = applyPatternsAndFoldGreedily(mod, *patterns, config);

    if (succeeded(converged)) return success();

    return success();
  }

  void runOnOperation() final {
    LLVM_DEBUG(llvm::dbgs() << "Start parallel ToucanCanonicalizer pass\n");
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

    LLVM_DEBUG(llvm::dbgs() << "Done parallel ToucanCanonicalizer pass\n");
    if (failed(result)) return signalPassFailure();
  }

};



std::unique_ptr<mlir::Pass> toucan::createToucanCanonicalizerPass() {
  return std::make_unique<ToucanCanonicalizerPass>();
}

std::unique_ptr<mlir::Pass> toucan::createToucanCanonicalizerPass(const GreedyRewriteConfig &config) {
  return std::make_unique<ToucanCanonicalizerPass>(config);
}
