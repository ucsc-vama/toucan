
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"

#include <memory>
#include <atomic>


#define GEN_PASS_DEF_REMOVEMEMREADEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "RemoveMemReadEnPass"

static std::atomic<uint64_t> memsWithEnInModules;
static std::atomic<uint64_t> memsWithoutEnInModules;

struct FirMemReadRewrite: OpRewritePattern<seq::FirMemReadOp> {
  using OpRewritePattern<seq::FirMemReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(seq::FirMemReadOp op, PatternRewriter &rewriter) const final {
    // Common info
    auto opMem = op.getMemory();
    auto opAddr = op.getAddress();
    auto clock = op.getClk();

    auto opEn = op.getEnable();

    if (opEn) {
      // has enable
      memsWithEnInModules++;
      
      // For now, simply remove Enable signal.
      auto newReadOp = rewriter.create<seq::FirMemReadOp>(op.getLoc(), opMem, opAddr, clock, nullptr);
      copyCustomizedAttrs(op, newReadOp);
      rewriter.replaceOp(op, newReadOp);
      return success();
    } else {
      // do nothing
      memsWithoutEnInModules++;
    }

    return failure();
  }
};


struct RemoveMemReadEnPass : toucan::impl::RemoveMemReadEnBase<RemoveMemReadEnPass> {
  using RemoveMemReadEnBase<RemoveMemReadEnPass>::RemoveMemReadEnBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {
    memsWithEnInModules = 0;
    memsWithoutEnInModules = 0;

    RewritePatternSet owningPatterns(context);
    owningPatterns.add<FirMemReadRewrite>(context);
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

    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result)) {
      // Don't care if failed
      ;
    }

    memsWithEn = memsWithEnInModules;
    memsWithoutEn = memsWithoutEnInModules;
  }
};

std::unique_ptr<mlir::Pass> toucan::createRemoveMemReadEnPass() {
  return std::make_unique<RemoveMemReadEnPass>();
}
