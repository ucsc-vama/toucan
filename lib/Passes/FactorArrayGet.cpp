
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
#include "mlir/IR/Value.h"
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
#include <atomic>


#define GEN_PASS_DEF_FACTORARRAYGETMUX
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "FactorArrayGetMuxPass"

static std::atomic<uint64_t> arrayMuxLoweredInModules;
static std::atomic<uint64_t> arrayConcatLoweredInModules;

struct LowerArrayMux: OpRewritePattern<comb::MuxOp> {
  using OpRewritePattern<comb::MuxOp>::OpRewritePattern;

  static bool arrayIsDirectlyDefined(Operation *arrayDefiningOp) {
    if (isa<hw::ArrayCreateOp>(arrayDefiningOp) || isa<hw::AggregateConstantOp>(arrayDefiningOp)) {
      // Do nothing for direct array access
      return true;
    }
    return false;
  }

  void getVecElmVals(PatternRewriter &rewriter, mlir::Operation *valOp, mlir::SmallVector<mlir::Value> &elms) const {
    elms.clear();
    if (auto op = dyn_cast<hw::ArrayCreateOp>(valOp)) {
      elms = op.getInputs();
    } else {
      assert(isa<hw::AggregateConstantOp>(valOp));

      for (auto eachConst: cast<hw::AggregateConstantOp>(valOp).getFields().getValue()) {
        auto constIntAttr = eachConst.cast<mlir::IntegerAttr>();
        auto constOp = rewriter.create<hw::ConstantOp>(valOp->getLoc(), constIntAttr);
        elms.push_back(constOp.getResult());
      }
    }
  }

  LogicalResult matchAndRewrite(comb::MuxOp muxOp, PatternRewriter &rewriter) const final {
    auto tValOp = muxOp.getTrueValue().getDefiningOp();
    auto fValOp = muxOp.getFalseValue().getDefiningOp();

    if (tValOp == nullptr || fValOp == nullptr) {
      return failure();
    }

    auto muxCond = muxOp.getCond();

    if (arrayIsDirectlyDefined(tValOp) && arrayIsDirectlyDefined(fValOp)) {
      // do something

      mlir::SmallVector<mlir::Value> resultVecElms, tVecElms, fVecElms;

      getVecElmVals(rewriter, tValOp, tVecElms);
      getVecElmVals(rewriter, fValOp, fVecElms);

      assert(tVecElms.size() == fVecElms.size());
      assert(tVecElms.size() != 0);

      for (size_t i = 0; i < tVecElms.size(); i++) {
        auto elemMuxOp = rewriter.create<comb::MuxOp>(muxOp->getLoc(), muxCond, tVecElms[i], fVecElms[i]);
        resultVecElms.push_back(elemMuxOp.getResult());
      }

      auto defVecOp = rewriter.create<hw::ArrayCreateOp>(muxOp.getLoc(), resultVecElms);
      muxOp->replaceAllUsesWith(defVecOp);

      return success();
    }

    return failure();
  }
};


struct ExpandVectorConcat: OpRewritePattern<hw::ArrayConcatOp> {
  using OpRewritePattern<hw::ArrayConcatOp>::OpRewritePattern;

  LogicalResult getArrayElementsFromConcat(hw::ArrayConcatOp op, PatternRewriter &rewriter, mlir::SmallVector<mlir::Value> &newVecElems) const {
    for (const auto &eachArrayVal: op.getInputs()) {
      assert(isa<hw::ArrayType>(eachArrayVal.getType()));

      auto valDefiningOp = eachArrayVal.getDefiningOp();

      if (auto arrayCreateOp = dyn_cast<hw::ArrayCreateOp>(valDefiningOp)) {
        for (auto eachVal: arrayCreateOp.getInputs()) {
          newVecElems.push_back(eachVal);
        }
      } else if (auto constArrayOp = dyn_cast<hw::AggregateConstantOp>(valDefiningOp)) {
        // const vec
        for (auto eachConst: constArrayOp.getFields().getValue()) {
          auto constIntAttr = eachConst.cast<mlir::IntegerAttr>();
          auto constOp = rewriter.create<hw::ConstantOp>(op->getLoc(), constIntAttr);
          newVecElems.push_back(constOp.getResult());
        }
      } else if (auto valDefiningConcatOp = dyn_cast<hw::ArrayConcatOp>(valDefiningOp)) {
        auto ret = getArrayElementsFromConcat(valDefiningConcatOp, rewriter, newVecElems);
        if (failed(ret)) return ret;
      } else {
        return failure();
        // valDefiningOp->print(llvm::errs());
        // llvm::errs() << "\n";
        // llvm_unreachable("Unsupported val producing a vector value");
      }
    }

    return success();
  }

  LogicalResult matchAndRewrite(hw::ArrayConcatOp op, PatternRewriter &rewriter) const final {
    mlir::SmallVector<mlir::Value> newVecElems;
    auto ret = getArrayElementsFromConcat(op, rewriter, newVecElems);
    if (failed(ret)) return ret;

    auto newArrayDefOp = rewriter.create<hw::ArrayCreateOp>(op->getLoc(), newVecElems);

    op->replaceAllUsesWith(newArrayDefOp);
    op.erase();

    arrayConcatLoweredInModules++;

    return success();
  }
};


struct FactorArrayGetMuxPass : toucan::impl::FactorArrayGetMuxBase<FactorArrayGetMuxPass> {
  using FactorArrayGetMuxBase<FactorArrayGetMuxPass>::FactorArrayGetMuxBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    owningPatterns.add<LowerArrayMux>(context);
    owningPatterns.add<ExpandVectorConcat>(context);
    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }


  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    return applyPatternsAndFoldGreedily(mod, *patterns);
  }

  void runOnOperation() final {
    auto mod = getOperation();

    arrayMuxLoweredInModules = 0;
    arrayConcatLoweredInModules = 0;

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

    arrayMuxLowered = arrayMuxLoweredInModules;
    arrayConcatLowered = arrayConcatLoweredInModules;
  }

};

std::unique_ptr<mlir::Pass> toucan::createFactorArrayGetMuxPass() {
  return std::make_unique<FactorArrayGetMuxPass>();
}
