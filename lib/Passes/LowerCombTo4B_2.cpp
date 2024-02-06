
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


#define GEN_PASS_DEF_LOWERCOMBTO4B_2
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombTo4B_2Pass"


struct AddSubCore {
  public:
  Value addCore(Operation *op, PatternRewriter &rewriter, Value lhsValue, Value rhsValue, Value base_carry) const {
    auto lhsValues = split_value_4B(op, lhsValue, rewriter);
    auto rhsValues = split_value_4B(op, rhsValue, rewriter);

    SmallVector<Value> results;
    auto last_carry = base_carry;
    for (size_t i = 0; i < lhsValues.size(); i++) {
      auto lhs = lhsValues[i];
      auto rhs = rhsValues[i];

      auto addOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Add, last_carry, lhs, rhs);
      auto addValue = addOp.getResult();
      results.push_back(addValue);

      if (i < (lhsValues.size() - 1)) {
        // Not last one, generate carry signal
        auto carryOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Carry, lhs, rhs);
        auto carryValue = carryOp.getResult();
        last_carry = carryValue;
      }
    }

    // Concat
    auto concatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), results);
    
    return concatOp.getResult();
  }
};



struct LowerCombAddOp: OpRewritePattern<comb::AddOp>, AddSubCore {
  using OpRewritePattern<comb::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AddOp op, PatternRewriter &rewriter) const final {
    auto const1BOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getI1Type(), 0);
    auto constZero1B = const1BOp.getResult();

    auto inputs = op.getInputs();
    if (inputs.size() != 2) {
      op.emitError() << "Expect exactly 2 operands but got " << inputs.size();
      return failure();
    }

    auto lhs = inputs[0];
    auto rhs = inputs[1];

    auto addResult = addCore(op.getOperation(), rewriter, lhs, rhs, constZero1B);

    copyCustomizedAttrs(op, addResult.getDefiningOp());
    rewriter.replaceOp(op, addResult);
    
    return success();
  }
};

struct LowerCombSubOp: OpRewritePattern<comb::SubOp>, AddSubCore {
  using OpRewritePattern<comb::SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::SubOp op, PatternRewriter &rewriter) const final {
    auto const1BOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getI1Type(), 1);
    auto constOne1B = const1BOp.getResult();

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto rhsValueWidth = hw::getBitWidth(rhs.getType());
    auto constOneOp = rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(rhsValueWidth, -1, true));
    auto notRhsOp = rewriter.create<comb::XorOp>(op.getLoc(), ValueRange({rhs, constOneOp.getResult()}), false);

    auto notRhs = notRhsOp.getResult();

    auto addResult = addCore(op.getOperation(), rewriter, lhs, notRhs, constOne1B);

    copyCustomizedAttrs(op, addResult.getDefiningOp());
    rewriter.replaceOp(op, addResult);
    
    return success();
  }
};

struct LowerCombMuxOp: OpRewritePattern<comb::MuxOp> {
  using OpRewritePattern<comb::MuxOp>::OpRewritePattern;

  // TODO: This is too slow

  LogicalResult matchAndRewrite(comb::MuxOp op, PatternRewriter &rewriter) const final {
    auto condVal = op.getCond();
    auto tValValue = op.getTrueValue();
    auto fValValue = op.getFalseValue();

    auto valueWidth = hw::getBitWidth(tValValue.getType());
    if (valueWidth > 4) {
      // return failure();
      auto tValValues = split_value_4B(op.getOperation(), tValValue, rewriter);
      auto fValValues = split_value_4B(op.getOperation(), fValValue, rewriter);

      SmallVector<Value, 256> results;
      for (auto [tval, fval]: zip(tValValues, fValValues)) {

        auto muxOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Mux, condVal, tval, fval);
        results.push_back(muxOp.getResult());
      }

      auto concatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), results);


      copyCustomizedAttrs(op, concatOp);
      rewriter.replaceOp(op, concatOp);
    } else {
      auto muxOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Mux, condVal, tValValue, fValValue);

      copyCustomizedAttrs(op, muxOp);
      rewriter.replaceOp(op, muxOp);
    }
    
    return success();
  }
};






struct LowerCombTo4B_2Pass : toucan::impl::LowerCombTo4B_2Base<LowerCombTo4B_2Pass> {
  using LowerCombTo4B_2Base<LowerCombTo4B_2Pass>::LowerCombTo4B_2Base;

  std::shared_ptr<FrozenRewritePatternSet> patterns;
  std::shared_ptr<ConversionTarget> target;

  LogicalResult initialize(MLIRContext *context) override {

    RewritePatternSet owningPatterns(context);
    ConversionTarget conversionTarget(*context);
    
    owningPatterns.add<LowerCombAddOp>(context);
    owningPatterns.add<LowerCombSubOp>(context);
    owningPatterns.add<LowerCombMuxOp>(context);


    conversionTarget.addLegalDialect<toucan::ToucanDialect>();
    conversionTarget.addLegalDialect<hw::HWDialect>();
    conversionTarget.addLegalDialect<comb::CombDialect>();

    // After lowering, following ops should no longer appear
    conversionTarget.addIllegalOp<comb::AddOp>();
    conversionTarget.addIllegalOp<comb::SubOp>();
    conversionTarget.addIllegalOp<comb::MuxOp>();
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

std::unique_ptr<mlir::Pass> toucan::createLowerCombTo4B_2Pass() {
  return std::make_unique<LowerCombTo4B_2Pass>();
}
