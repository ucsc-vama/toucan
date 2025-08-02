
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "toucan/ToucanVecOpLimits.h"

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
#include <atomic>
#include <mutex>


#define GEN_PASS_DEF_LOWERCOMBTO4B_2
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombTo4B_2Pass"

static std::atomic<uint64_t> numCombAddInModules;
static std::atomic<uint64_t> numCombSubInModules;
static std::atomic<uint64_t> numCombMuxInModules;


struct AddSubCore {
  public:
  enum AddOrSub {
    Add,
    Sub
  };


  Value addSubCore(AddOrSub addOrSub, Operation *op, PatternRewriter &rewriter, Value lhsValue, Value rhsValue, std::optional<StringAttr> namehint) const {
    auto inputValWidth = hw::getBitWidth(lhsValue.getType());
    assert(hw::getBitWidth(rhsValue.getType()) == inputValWidth);
    assert(inputValWidth <= TOUCAN_VEC_OP_MAX_WIDTH);

    if (inputValWidth <= 4) {
      // small add. Simply use an add LUT
      LUTOpName opName;
      if (addOrSub == AddOrSub::Add) {
        opName = LUTOpName::LUT_Add;
      } else {
        assert(addOrSub == AddOrSub::Sub);
        opName = LUTOpName::LUT_Sub;
      }

      auto lutOp = rewriter.create<toucan::LUTOp>(op->getLoc(), opName, lhsValue, rhsValue);
      auto result = lutOp.getResult();

      if (inputValWidth != 4) {
        auto extractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), lutOp.getResult(), 0, inputValWidth);
        result = extractOp.getResult();
      }

      if (namehint) {
        setSVNameHintAttr(result.getDefiningOp(), namehint.value());
      }

      return result;
    } else {
      // large value. use vector op

      auto lhsPadding = padding_with_0_and_align_4b(op, rewriter, lhsValue);
      auto rhsPadding = padding_with_0_and_align_4b(op, rewriter, rhsValue);

      auto lhsValues = split_value_4B(op, lhsPadding, rewriter);
      auto rhsValues = split_value_4B(op, rhsPadding, rewriter);

      auto lhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op->getLoc(), lhsValues);
      auto rhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op->getLoc(), rhsValues);

      auto lhsVecHandle = lhsVecDeclOp.getHandle();
      auto rhsVecHandle = rhsVecDeclOp.getHandle();

      VecArithOpName opName;
      if (addOrSub == AddOrSub::Add) {
        opName = VecArithOpName::VecArith_Add;
      } else {
        assert(addOrSub == AddOrSub::Sub);
        opName = VecArithOpName::VecArith_Sub;
      }

      auto vecOp = rewriter.create<toucan::VectorArithOp>(op->getLoc(), opName, lhsVecHandle, rhsVecHandle);
      auto vecAddResultVec = vecOp.getResult();

      // attachNameHintAndFragmentId(rewriter, results, namehint);

      auto resultVal = convertFullVectorBackToValue(rewriter, op->getLoc(), vecAddResultVec);

      auto resultValWidth = hw::getBitWidth(resultVal.getType());
      if (inputValWidth != resultValWidth) {
        assert(inputValWidth < resultValWidth);
        // Need clear top bits
        auto extractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), resultVal, 0, inputValWidth);
        resultVal = extractOp.getResult();
      }

      return resultVal;
    }
  }

};



struct LowerCombAddOp: OpRewritePattern<comb::AddOp>, AddSubCore {
  using OpRewritePattern<comb::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AddOp op, PatternRewriter &rewriter) const final {

    auto inputs = op.getInputs();
    if (inputs.size() != 2) {
      op.emitError() << "Expect exactly 2 operands but got " << inputs.size();
      return failure();
    }
    numCombAddInModules++;

    auto lhs = inputs[0];
    auto rhs = inputs[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));
    auto inputValWidth = hw::getBitWidth(lhs.getType());

    if (inputValWidth <= TOUCAN_VEC_OP_MAX_WIDTH) {
      auto optionalNameHint = getSVNameHintAttr(op);;
      auto addResult = addSubCore(AddOrSub::Add, op.getOperation(), rewriter, lhs, rhs, optionalNameHint);

      assert(hw::getBitWidth(addResult.getType()) == hw::getBitWidth(lhs.getType()));

      rewriter.replaceOp(op, addResult);
    } else {
      // TODO: Need more testing
      int maxSectionInEachSlice = TOUCAN_VEC_OP_MAX_SECTIONS - 1;
      // use multiple add, highest section is used as carry bit
      int maxBitsInEachSlice = maxSectionInEachSlice * 4;

      auto constZero4BVal = rewriter.create<hw::ConstantOp>(op->getLoc(), rewriter.getI4Type(), 0).getResult();
      mlir::SmallVector<mlir::Value> resultSections;
      mlir::Value carryVal = constZero4BVal;

      for (int remainingBits = inputValWidth; remainingBits > 0; remainingBits -= maxBitsInEachSlice) {
        int bitsInThisSlice = std::min(maxBitsInEachSlice, remainingBits);
        int startingBitPos = inputValWidth - remainingBits;

        assert(startingBitPos >= 0);
        auto newlhs = rewriter.create<comb::ExtractOp>(op.getLoc(), lhs, startingBitPos, bitsInThisSlice).getResult();
        auto newrhs = rewriter.create<comb::ExtractOp>(op.getLoc(), rhs, startingBitPos, bitsInThisSlice).getResult();

        auto newPaddingLhs = rewriter.create<comb::ConcatOp>(op->getLoc(), constZero4BVal, newlhs).getResult();
        auto newPaddingRhs = rewriter.create<comb::ConcatOp>(op->getLoc(), constZero4BVal, newrhs).getResult();

        auto addResult = addSubCore(AddOrSub::Add, op.getOperation(), rewriter, newPaddingLhs, newPaddingRhs, nullptr);
        if (carryVal != constZero4BVal) {
          auto padding = rewriter.create<hw::ConstantOp>(op->getLoc(), rewriter.getIntegerType(bitsInThisSlice), 0).getResult();
          auto fullBitCarry = rewriter.create<comb::ConcatOp>(op->getLoc(), padding, carryVal).getResult();
          addResult = addSubCore(AddOrSub::Add, op.getOperation(), rewriter, addResult, fullBitCarry, nullptr);
        }

        auto resultSectionVal = rewriter.create<comb::ExtractOp>(op->getLoc(), addResult, 0, bitsInThisSlice).getResult();
        carryVal = rewriter.create<comb::ExtractOp>(op->getLoc(), addResult, bitsInThisSlice, 4).getResult();
        assert(hw::getBitWidth(resultSectionVal.getType()) == bitsInThisSlice);

        resultSections.push_back(resultSectionVal);
      }

      std::reverse(resultSections.begin(), resultSections.end());

      auto optionalNameHint = getSVNameHintAttr(op);;
      auto addResult = rewriter.create<comb::ConcatOp>(op->getLoc(), resultSections).getResult();
      attachNameHintAndFragmentId(rewriter, addResult, optionalNameHint);

      assert(hw::getBitWidth(addResult.getType()) == hw::getBitWidth(lhs.getType()));

      rewriter.replaceOp(op, addResult);
    }

    return success();
  }
};

struct LowerCombSubOp: OpRewritePattern<comb::SubOp>, AddSubCore {
  using OpRewritePattern<comb::SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::SubOp op, PatternRewriter &rewriter) const final {
    numCombSubInModules++;

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));
    assert(hw::getBitWidth(rhs.getType()) < TOUCAN_VEC_OP_MAX_WIDTH);

    auto optionalNameHint = getSVNameHintAttr(op);;
    auto subResult = addSubCore(AddOrSub::Sub, op.getOperation(), rewriter, lhs, rhs, optionalNameHint);

    assert(hw::getBitWidth(subResult.getType()) == hw::getBitWidth(lhs.getType()));

    rewriter.replaceOp(op, subResult);

    return success();
  }
};

struct LowerCombMuxOp: OpRewritePattern<comb::MuxOp> {
  using OpRewritePattern<comb::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::MuxOp op, PatternRewriter &rewriter) const final {
    numCombMuxInModules++;

    auto condVal = op.getCond();
    auto tValValue = op.getTrueValue();
    auto fValValue = op.getFalseValue();
    auto valueWidth = hw::getBitWidth(tValValue.getType());
    auto optionalNameHint = getSVNameHintAttr(op);

    if (valueWidth > 4) {
      // return failure();
      auto tValValues = split_value_4B(op.getOperation(), tValValue, rewriter);
      auto fValValues = split_value_4B(op.getOperation(), fValValue, rewriter);

      SmallVector<Value> results;
      for (auto [tval, fval]: zip(tValValues, fValValues)) {

        auto muxOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Mux, condVal, tval, fval);
        results.push_back(muxOp.getResult());
      }

      attachNameHintAndFragmentId(rewriter, results, optionalNameHint);

      auto concatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), results);

      rewriter.replaceOp(op, concatOp);
    } else {
      auto muxOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Mux, condVal, tValValue, fValValue);

      attachNameHintAndFragmentId(rewriter, muxOp, optionalNameHint);
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
    numCombAddInModules = 0;
    numCombSubInModules = 0;
    numCombMuxInModules = 0;


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

    numCombAdd = numCombAddInModules;
    numCombSub = numCombSubInModules;
    numCombMux = numCombMuxInModules;
  }

};

std::unique_ptr<mlir::Pass> toucan::createLowerCombTo4B_2Pass() {
  return std::make_unique<LowerCombTo4B_2Pass>();
}
