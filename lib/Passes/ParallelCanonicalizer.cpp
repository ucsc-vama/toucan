
#include "circt/Dialect/HW/HWDialect.h"
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>


#define GEN_PASS_DEF_PARALLELCANONICALIZER
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "ParallelCanonicalizerPass"





 #include "mlir/Transforms/Passes.h"
  
 #include "mlir/Pass/Pass.h"
 #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
  
 namespace mlir {
 #define GEN_PASS_DEF_CANONICALIZER
 #include "mlir/Transforms/Passes.h.inc"
 } // namespace mlir
  
 using namespace mlir;
  





struct ParallelCanonicalizerPass : toucan::impl::ParallelCanonicalizerBase<ParallelCanonicalizerPass> {
  using ParallelCanonicalizerBase<ParallelCanonicalizerPass>::ParallelCanonicalizerBase;

  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;

  ArrayRef<std::string> disabledPatterns;
  ArrayRef<std::string> enabledPatterns;

  ParallelCanonicalizerPass() = default;
  ParallelCanonicalizerPass(const GreedyRewriteConfig &config)
      : config(config) { }

  ParallelCanonicalizerPass(const GreedyRewriteConfig &config,
                ArrayRef<std::string> disabledPatterns,
                ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }



  LogicalResult initialize(MLIRContext *context) override {

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects()) {
      if (isa<comb::CombDialect>(dialect) || isa<toucan::ToucanDialect>(dialect));
      dialect->getCanonicalizationPatterns(owningPatterns);
    }
      
    for (RegisteredOperationName op : context->getRegisteredOperations()) {
      // llvm::dbgs() << op << "\n";
      // if (enabledOps.contains(op.getStringRef())) {

      // }
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



//  /// Create a Canonicalizer pass.
//  std::unique_ptr<Pass> mlir::createCanonicalizerPass() {
//    return std::make_unique<Canonicalizer>();
//  }
  
//  /// Creates an instance of the Canonicalizer pass with the specified config.
//  std::unique_ptr<Pass>
//  mlir::createCanonicalizerPass(const GreedyRewriteConfig &config,
//                                ArrayRef<std::string> disabledPatterns,
//                                ArrayRef<std::string> enabledPatterns) {
//    return std::make_unique<Canonicalizer>(config, disabledPatterns,
//                                           enabledPatterns);
//  }


std::unique_ptr<mlir::Pass> toucan::createParallelCanonicalizerPass() {
  return std::make_unique<ParallelCanonicalizerPass>();
}

std::unique_ptr<mlir::Pass> toucan::createParallelCanonicalizerPass(const GreedyRewriteConfig &config) {
  return std::make_unique<ParallelCanonicalizerPass>(config);
}
