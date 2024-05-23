
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <memory>
#include <atomic>


#define GEN_PASS_DEF_FACTORBINARYOP
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "FactorBinaryOpPass"


static std::atomic<uint64_t> numMultiOprandBinOpInModule;


struct LowerBinaryOpWithMultipleOperandBase {
  public:
  template<class OpTy>
  LogicalResult lowerOp(OpTy &op, PatternRewriter &rewriter) const {
    auto inputs = op.getInputs();
    if (inputs.size() > 2) {
      numMultiOprandBinOpInModule++;

      SmallVector<Value> inputs_vec;
      inputs_vec.assign(inputs.begin(), inputs.end());
      auto resultVal = generate_reduce_tree(rewriter, op.getLoc(), inputs_vec, [&](RewriterBase &rewriter, Location loc, Value lhs, Value rhs) {
        auto newOp = rewriter.create<OpTy>(loc, ValueRange({lhs, rhs}), false);
        return newOp.getResult();
      });

      auto namehint = getSVNameHintAttr(op);
      if (namehint) {
        setSVNameHintAttr(resultVal.getDefiningOp(), namehint.value());
      }

      rewriter.replaceOp(op, resultVal);
    }
    return success();
  }
};


struct LowerCombAndWithMultipleOperands: OpRewritePattern<comb::AndOp>, LowerBinaryOpWithMultipleOperandBase {
  using OpRewritePattern<comb::AndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AndOp op, PatternRewriter &rewriter) const final {
    return lowerOp<comb::AndOp>(op, rewriter);
  }
};

struct LowerCombOrWithMultipleOperands: OpRewritePattern<comb::OrOp>, LowerBinaryOpWithMultipleOperandBase {
  using OpRewritePattern<comb::OrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::OrOp op, PatternRewriter &rewriter) const final {
    return lowerOp<comb::OrOp>(op, rewriter);
  }
};

struct LowerCombXorWithMultipleOperands: OpRewritePattern<comb::XorOp>, LowerBinaryOpWithMultipleOperandBase {
  using OpRewritePattern<comb::XorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::XorOp op, PatternRewriter &rewriter) const final {
    return lowerOp<comb::XorOp>(op, rewriter);
  }
};

struct LowerCombAddWithMultipleOperands: OpRewritePattern<comb::AddOp>, LowerBinaryOpWithMultipleOperandBase {
  using OpRewritePattern<comb::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AddOp op, PatternRewriter &rewriter) const final {
    return lowerOp<comb::AddOp>(op, rewriter);
  }
};




struct FactorBinaryOpPass : toucan::impl::FactorBinaryOpBase<FactorBinaryOpPass> {
  using FactorBinaryOpBase<FactorBinaryOpPass>::FactorBinaryOpBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;
  LogicalResult initialize(MLIRContext *context) override {
    numMultiOprandBinOpInModule = 0;

    RewritePatternSet owningPatterns(context);
    
    owningPatterns.add<LowerCombAddWithMultipleOperands>(context);
    owningPatterns.add<LowerCombAndWithMultipleOperands>(context);
    owningPatterns.add<LowerCombOrWithMultipleOperands>(context);
    owningPatterns.add<LowerCombXorWithMultipleOperands>(context);

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));

    return success();
  }


  LogicalResult runOnModule(hw::HWModuleOp mod) {
    GreedyRewriteConfig config;
    config.maxIterations = 10;
    config.useTopDownTraversal = true;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    auto ret = applyPatternsAndFoldGreedily(mod, *patterns, config);
    // This pass won't convert all matched operations. It's fine to fail
    if (failed(ret)) {
      ;
    }
    return success();
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

    numMultiOprandBinaryOp = numMultiOprandBinOpInModule;
  }
};

std::unique_ptr<mlir::Pass> toucan::createFactorBinaryOpPass() {
  return std::make_unique<FactorBinaryOpPass>();
}
