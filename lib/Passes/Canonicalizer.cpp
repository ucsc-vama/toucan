
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


#define GEN_PASS_DEF_CANONICALIZER
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "CanonicalizerPass"




struct CanonicalizerPass : toucan::impl::CanonicalizerBase<CanonicalizerPass> {
  using CanonicalizerBase<CanonicalizerPass>::CanonicalizerBase;

  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;

  ArrayRef<std::string> disabledPatterns;
  ArrayRef<std::string> enabledPatterns;

  CanonicalizerPass() = default;
  CanonicalizerPass(const GreedyRewriteConfig &config)
      : config(config) { }

  CanonicalizerPass(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  LogicalResult initialize(MLIRContext *context) override {

    RewritePatternSet owningPatterns(context);
    // Don't run canonicalizer for hw dialect!
    for (auto *dialect : context->getLoadedDialects()) {
      if (isa<comb::CombDialect>(dialect)
        || isa<toucan::ToucanDialect>(dialect)
        || isa<seq::SeqDialect>(dialect)) {
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
    LLVM_DEBUG(llvm::dbgs() << "Start parallel canonicalizer pass\n");
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

    LLVM_DEBUG(llvm::dbgs() << "Done parallel canonicalizer pass\n");
    if (failed(result)) return signalPassFailure();
  }

};



std::unique_ptr<mlir::Pass> toucan::createCanonicalizerPass() {
  return std::make_unique<CanonicalizerPass>();
}

std::unique_ptr<mlir::Pass> toucan::createCanonicalizerPass(const GreedyRewriteConfig &config) {
  return std::make_unique<CanonicalizerPass>(config);
}
