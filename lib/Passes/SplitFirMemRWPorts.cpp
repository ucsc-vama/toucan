
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


#define GEN_PASS_DEF_SPLITFIRMEMRWPORTS
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "SplitFirMemRWPortsPass"

std::atomic<uint64_t> splittedRWPortsInModules;

struct FirMemRWPortRewrite: OpRewritePattern<seq::FirMemReadWriteOp> {
  using OpRewritePattern<seq::FirMemReadWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(seq::FirMemReadWriteOp op, PatternRewriter &rewriter) const final {
    // Common info
    auto opMem = op.getMemory();
    auto opAddr = op.getAddress();
    auto opEn = op.getEnable();
    // mode == 1 => write, mode == 0 => read
    auto opMode = op.getMode();
    auto clock = op.getClk();

    // write
    auto opWriteData = op.getWriteData();
    auto opWriteMask = op.getMask();

    // // read
    // auto opReadData = op.getReadData();

    // read_en
    auto constTrue = rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(1, 1));
    auto isModeReadOp = rewriter.create<comb::XorOp>(op.getLoc(), opMode, constTrue);
    auto readEnOp = rewriter.create<comb::AndOp>(op->getLoc(), isModeReadOp, opEn);
    // read op
    auto readOp = rewriter.create<seq::FirMemReadOp>(op->getLoc(), opMem, opAddr, clock, readEnOp);
    // write op
    auto writeEnOp = rewriter.create<comb::AndOp>(op->getLoc(), opMode, opEn);
    rewriter.create<seq::FirMemWriteOp>(op.getLoc(), opMem, opAddr, clock, writeEnOp, opWriteData, opWriteMask);

    rewriter.replaceOp(op, readOp);

    splittedRWPortsInModules++;

    return success();
  }
};




struct SplitFirMemRWPortsPass : toucan::impl::SplitFirMemRWPortsBase<SplitFirMemRWPortsPass> {
  using SplitFirMemRWPortsBase<SplitFirMemRWPortsPass>::SplitFirMemRWPortsBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {
    splittedRWPortsInModules = 0;

    RewritePatternSet owningPatterns(context);
    owningPatterns.add<FirMemRWPortRewrite>(context);
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
    if (failed(result)) return signalPassFailure();

    splittedRWPorts = splittedRWPortsInModules;
  }


};

std::unique_ptr<mlir::Pass> toucan::createSplitFirMemRWPortsPass() {
  return std::make_unique<SplitFirMemRWPortsPass>();
}
