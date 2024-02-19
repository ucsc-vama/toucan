
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

#include <memory>
#include <string>


#define GEN_PASS_DEF_LOWERCOMBTO4B_3
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombTo4B_3Pass"


struct LowerCombAndOp: OpRewritePattern<comb::AndOp> {
  using OpRewritePattern<comb::AndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AndOp op, PatternRewriter &rewriter) const final {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
      return failure();
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    auto optionalNameHint = getSVNameHintAttr(op);

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_And, lhs, rhs);
        intermediateResults.push_back(andOp.getResult());
      }

      attachNameHintAndFragmentId(rewriter, intermediateResults, optionalNameHint);
      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_And, lhs, rhs);

      attachNameHintAndFragmentId(rewriter, andOp, optionalNameHint);
      rewriter.replaceOp(op, andOp);
    }
    
    return success();
  }
};


struct LowerCombOrOp: OpRewritePattern<comb::OrOp> {
  using OpRewritePattern<comb::OrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::OrOp op, PatternRewriter &rewriter) const final {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
      return failure();
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());
    
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    auto optionalNameHint = getSVNameHintAttr(op);

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto orOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Or, lhs, rhs);
        intermediateResults.push_back(orOp.getResult());
      }
      
      attachNameHintAndFragmentId(rewriter, intermediateResults, optionalNameHint);
      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto orOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Or, lhs, rhs);

      attachNameHintAndFragmentId(rewriter, orOp, optionalNameHint);
      rewriter.replaceOp(op, orOp);
    }
    
    return success();
  }
};


struct LowerCombXorOp: OpRewritePattern<comb::XorOp> {
  using OpRewritePattern<comb::XorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::XorOp op, PatternRewriter &rewriter) const final {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
      return failure();
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());
    
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    auto optionalNameHint = getSVNameHintAttr(op);

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto xorOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Xor, lhs, rhs);
        intermediateResults.push_back(xorOp.getResult());
      }

      attachNameHintAndFragmentId(rewriter, intermediateResults, optionalNameHint);
      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto xorOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Xor, lhs, rhs);

      attachNameHintAndFragmentId(rewriter, xorOp, optionalNameHint);
      rewriter.replaceOp(op, xorOp);
    }
    
    return success();
  }
};


// struct LowerHWConstantOp: OpRewritePattern<hw::ConstantOp> {
//   using OpRewritePattern<hw::ConstantOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(hw::ConstantOp op, PatternRewriter &rewriter) const final {
//     SmallVector<Value> results;

//     auto constValueWidth = op.getValue().getBitWidth();
//     // auto constValueRaw = op.getValue().extractBits(0, 2);

//     if (constValueWidth > 4) {
//       auto chunks = split_signal_4B(constValueWidth);
//       for (auto [chunkId, chunkWidth]: chunks) {
//         auto newValue = op.getValue().extractBits(chunkWidth, chunkId * 4);
//         auto newConstOp = rewriter.create<hw::ConstantOp>(op->getLoc(), newValue);
//         results.push_back(newConstOp.getResult());
//       }

//       auto bitConcatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), results);
//       copyCustomizedAttrs(op, bitConcatOp);
//       rewriter.replaceOp(op, bitConcatOp);
//       return success();
//     }
//     return failure();
//   }
// };




struct LowerCombTo4B_3Pass : toucan::impl::LowerCombTo4B_3Base<LowerCombTo4B_3Pass> {
  using LowerCombTo4B_3Base<LowerCombTo4B_3Pass>::LowerCombTo4B_3Base;

  std::shared_ptr<FrozenRewritePatternSet> patterns;
  std::shared_ptr<ConversionTarget> target;

  LogicalResult initialize(MLIRContext *context) override {

    RewritePatternSet owningPatterns(context);
    ConversionTarget conversionTarget(*context);
    
    // owningPatterns.add<LowerHWConstantOp>(context);
    owningPatterns.add<LowerCombAndOp>(context);
    owningPatterns.add<LowerCombOrOp>(context);
    owningPatterns.add<LowerCombXorOp>(context);

    conversionTarget.addLegalDialect<toucan::ToucanDialect>();
    conversionTarget.addLegalDialect<hw::HWDialect>();
    conversionTarget.addLegalDialect<comb::CombDialect>();

    // After lowering, following ops should no longer appear
    conversionTarget.addIllegalOp<comb::AndOp>();
    conversionTarget.addIllegalOp<comb::OrOp>();
    conversionTarget.addIllegalOp<comb::XorOp>();
    // conversionTarget.addIllegalOp<comb::AddOp>();
    // conversionTarget.addIllegalOp<comb::SubOp>();
    // conversionTarget.addIllegalOp<comb::MuxOp>();
    // conversionTarget.addIllegalOp<comb::ReplicateOp>();
    // conversionTarget.addIllegalOp<comb::ShlOp>();
    // conversionTarget.addIllegalOp<comb::ShrUOp>();
    // conversionTarget.addIllegalOp<comb::ShrSOp>();
    // conversionTarget.addIllegalOp<comb::ICmpOp>();
    // conversionTarget.addIllegalOp<comb::MulOp>();

    // conversionTarget.addIllegalOp<comb::DivUOp>();
    // conversionTarget.addIllegalOp<comb::DivSOp>();
    // conversionTarget.addIllegalOp<comb::ModUOp>();
    // conversionTarget.addIllegalOp<comb::ModSOp>();

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    target = std::make_shared<ConversionTarget>(std::move(
    conversionTarget));

    return success();
  }


  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

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

std::unique_ptr<mlir::Pass> toucan::createLowerCombTo4B_3Pass() {
  return std::make_unique<LowerCombTo4B_3Pass>();
}
