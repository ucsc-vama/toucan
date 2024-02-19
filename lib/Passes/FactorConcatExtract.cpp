
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/PrettyStackTrace.h"

#include <memory>
#include <numeric>
#include <type_traits>


#define GEN_PASS_DEF_FACTORCONCATEXTRACT
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "FactorConcatExtractPass"


struct BitsOperations {

  static bool isElementsFullWidth(OperandRange &vals) {
    for (size_t i = 0; i < vals.size(); i++) {
      auto valBitWidth = hw::getBitWidth(vals[i].getType());
      if (i != 0) {
        if (valBitWidth != 4) return false;
      } else {
        if (valBitWidth > 4) return false;
      }
    } 
    return true;
  }

  static bool isElementsFullWidth(SmallVector<Value> &vals) {
    for (size_t i = 0; i < vals.size(); i++) {
      auto valBitWidth = hw::getBitWidth(vals[i].getType());
      if (i != 0) {
        if (valBitWidth != 4) return false;
      } else {
        if (valBitWidth > 4) return false;
      }
    } 
    return true;
  }

  static LUTOpName getShlNameUsingShamt(size_t shamt) {
    switch (shamt) {
      case 1: return LUTOpName::LUT_Shl1;
      case 2: return LUTOpName::LUT_Shl2;
      case 3: return LUTOpName::LUT_Shl3;
      default: llvm_unreachable("shl shamt out of range");
    }
  }

  static LUTOpName getShrNameUsingShamt(size_t shamt) {
    switch (shamt) {
      case 1: return LUTOpName::LUT_Shr1;
      case 2: return LUTOpName::LUT_Shr2;
      case 3: return LUTOpName::LUT_Shr3;
      default: llvm_unreachable("shr shamt out of range");
    }
  }

  static Value extractLowNBitsAndShiftToMSB(RewriterBase &rewriter, Location loc, Value val, Value constZero4B, size_t n) {
    auto inputWidth = static_cast<size_t>(hw::getBitWidth(val.getType()));
    assert(inputWidth <= 4);
    assert(n < inputWidth);
    assert(n > 0);

    size_t shamt = 4 - n;

    return bitsShlByNAndFill(rewriter, loc, val, constZero4B, shamt);
  }

  static Value bitsShlByNAndFill(RewriterBase &rewriter, Location loc, Value val, Value constZero4B, size_t shamt) {
    if (shamt == 0) return val;
    auto inputValWidth = hw::getBitWidth(val.getType());
    auto expectedWidth = std::min(4ULL, inputValWidth + shamt);
    auto resultType = rewriter.getIntegerType(expectedWidth);

    auto shlOp = rewriter.create<LUTOp>(loc, resultType, getShlNameUsingShamt(shamt), ValueRange({val, constZero4B}));
    return shlOp.getResult();
  }

  // static Value 
  static Value bitsShrByN(RewriterBase &rewriter, Location loc, Value lhs, Value rhs, size_t shamt) {
    if (shamt == 0) return rhs;
    auto lhsValWidth = static_cast<size_t>(hw::getBitWidth(lhs.getType()));
    auto rhsValWidth = static_cast<size_t>(hw::getBitWidth(rhs.getType()));
    assert(rhsValWidth > shamt);
    auto expectedWidth = std::min(4ul, lhsValWidth + rhsValWidth - shamt);
    // Here we know it's filled by 0s, so manually limit output width.
    auto resultType = rewriter.getIntegerType(expectedWidth);

    auto shlOp = rewriter.create<LUTOp>(loc, resultType, getShrNameUsingShamt(shamt), ValueRange({lhs, rhs}));
    return shlOp.getResult();
  }

  static Value bitsShrByNAndFill(RewriterBase &rewriter, Location loc, Value val, Value constZero4B, size_t shamt) {
    if (shamt == 0) return val;
    auto inputValWidth = static_cast<size_t>(hw::getBitWidth(val.getType()));
    assert(inputValWidth > shamt);
    auto expectedWidth = inputValWidth - shamt;
    // Here we know it's filled by 0s, so manually limit output width.
    auto resultType = rewriter.getIntegerType(expectedWidth);

    auto shlOp = rewriter.create<LUTOp>(loc, resultType, getShrNameUsingShamt(shamt), ValueRange({constZero4B, val}));
    return shlOp.getResult();
  }

  static Value extractHighNBitsAndShiftToLSB(RewriterBase &rewriter, Location loc, Value val, Value constZero4B, size_t n) {
    auto inputWidth = static_cast<size_t>(hw::getBitWidth(val.getType()));
    assert(inputWidth <= 4);
    if (n == inputWidth) return val;
    assert(n < inputWidth);

    size_t shamt = inputWidth - n;
    return bitsShrByNAndFill(rewriter, loc, val, constZero4B, shamt);
  }

  static Value concatBitByOr(RewriterBase &rewriter, Location loc, SmallVector<Value> &vals) {
    assert(!vals.empty());

    if (vals.size() == 1) return vals.front();
    auto elem_2nd = std::next(vals.begin());
    Value lastVal = *vals.begin();
    for (auto i = elem_2nd; i != vals.end(); i++) {
      auto val = *i;
      auto orOp = rewriter.create<toucan::LUTOp>(loc, toucan::LUTOpName::LUT_Or, lastVal, val);
      lastVal = orOp.getResult();
    }
    return lastVal;
  }

  static Value removeHighBits(RewriterBase &rewriter, Location loc, Value val, size_t bitsNeeded) {
    size_t valWidth = static_cast<size_t>(hw::getBitWidth(val.getType()));
    assert(valWidth <= 4);
    assert(bitsNeeded <= valWidth);
    if (bitsNeeded == valWidth) return val;

    size_t mask = (1 << bitsNeeded) - 1;

    auto maskConstOp = rewriter.create<hw::ConstantOp>(loc, APInt(valWidth, mask));
    auto maskConstVal = maskConstOp.getResult();

    // Note: Here we force the width of output to be bitsNeeded
    auto outputType = rewriter.getIntegerType(bitsNeeded);
    auto andOp = rewriter.create<LUTOp>(loc, outputType, LUTOpName::LUT_And, ValueRange({val, maskConstVal}));

    return andOp.getResult();
  }
};



struct LowerCombConcatOp: OpRewritePattern<comb::ConcatOp>, BitsOperations {
  using OpRewritePattern<comb::ConcatOp>::OpRewritePattern;


  LogicalResult handleNestedConcat(PatternRewriter &rewriter, comb::ConcatOp &op) const {
    auto catInputs = op.getInputs();
    auto outputValWidth = hw::getBitWidth(op.getResult().getType());

    // If any value inside catInputs is also a result of concat
    SmallVector<Value> newCatInputs;
    for (auto val: catInputs) {
      auto valDefiningOp = val.getDefiningOp();
      if (auto valCatOp = dyn_cast<comb::ConcatOp>(valDefiningOp)) {
        auto pCatInputs = valCatOp.getInputs();
        newCatInputs.append(pCatInputs.begin(), pCatInputs.end());
      } else {
        newCatInputs.push_back(val);
      }
    }
    auto newCatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), newCatInputs);
    copyCustomizedAttrs(op, newCatOp);
    assert(hw::getBitWidth(newCatOp.getResult().getType()) == outputValWidth);
    rewriter.replaceOp(op, newCatOp);

    return success();
  }

  LogicalResult alignInputs(PatternRewriter &rewriter, comb::ConcatOp &op) const {
    auto catInputs = op.getInputs();
    auto outputValWidth = hw::getBitWidth(op.getResult().getType());

    // Merge to seq of 4Bs
    auto loc = op.getLoc();

    auto constZeroOp = rewriter.create<hw::ConstantOp>(loc, rewriter.getI4Type(), 0);
    auto constZeroVal = constZeroOp.getResult();

    SmallVector<Value> alignedElems;

    size_t currentSegmentBits = 0;
    SmallVector<Value> currentSegments;

    for (auto val: llvm::reverse(catInputs)) {
      // do this from LSB
      auto valWidth = static_cast<size_t>(hw::getBitWidth(val.getType()));

      auto segmentBitsAfterConcat = currentSegmentBits + valWidth;
      if (segmentBitsAfterConcat > 4) {
        // Need split
        // low half belongs to current
        auto lowHalfBits = 4 - currentSegmentBits;
        auto lowHalfVal = extractLowNBitsAndShiftToMSB(rewriter, loc, val, constZeroVal, lowHalfBits);
        currentSegments.push_back(lowHalfVal);
        // high half belongs to next
        auto highHalfBits = valWidth - lowHalfBits;
        auto highHalfVal = extractHighNBitsAndShiftToLSB(rewriter, loc, val, constZeroVal, highHalfBits);
        // concat
        // std::reverse(currentSegments.begin(), currentSegments.end());
        auto result = concatBitByOr(rewriter, loc, currentSegments);
        alignedElems.push_back(result);
        // clear and push high bits
        currentSegments.clear();
        currentSegments.push_back(highHalfVal);
        currentSegmentBits = highHalfBits;
      } else {
        // simply push back
        size_t spacingAfterPush = 4 - segmentBitsAfterConcat;

        if (spacingAfterPush == 0) {
          // Last val of current 4b segment
          auto shiftResult = bitsShlByNAndFill(rewriter, loc, val, constZeroVal, currentSegmentBits % 4);
          currentSegments.push_back(shiftResult);
          // ends current segment
          // std::reverse(currentSegments.begin(), currentSegments.end());
          auto result = concatBitByOr(rewriter, loc, currentSegments);
          alignedElems.push_back(result);
          // reset for next section
          currentSegments.clear();
          currentSegmentBits = 0;
        } else {
          auto shiftResult = bitsShlByNAndFill(rewriter, loc, val, constZeroVal, currentSegmentBits % 4);
          currentSegments.push_back(shiftResult);
          currentSegmentBits += valWidth;
        }
      }
    }

    // concat heading
    if (!currentSegments.empty()) {
      auto result = concatBitByOr(rewriter, loc, currentSegments);
      alignedElems.push_back(result);
    }

    std::reverse(alignedElems.begin(), alignedElems.end());

    // Then create new op. copy attrs
    assert(!alignedElems.empty());
    if (alignedElems.size() > 1) {
      auto newCatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), alignedElems);
      copyCustomizedAttrs(op, newCatOp);
      assert(hw::getBitWidth(newCatOp.getResult().getType()) == outputValWidth);
      rewriter.replaceOp(op, newCatOp);
    } else {
      copyCustomizedAttrs(op, alignedElems[0].getDefiningOp());
      auto resultValWidth = hw::getBitWidth(alignedElems[0].getType());
      if (resultValWidth != outputValWidth) {
        emitError(loc) << "real " << resultValWidth <<", expect " << outputValWidth << "\n";
      }
      assert(resultValWidth == outputValWidth);
      rewriter.replaceOp(op, alignedElems[0]);
    }

    return success();
  }

  LogicalResult matchAndRewrite(comb::ConcatOp op, PatternRewriter &rewriter) const final {
    // Remove unused
    if (op->getUsers().empty()) {
      // No body using. Remove
      rewriter.eraseOp(op);
      return success();
    }

    auto catInputs = op.getInputs();

    if (catInputs.size() == 1) {
      // This shouldn't happen
      rewriter.replaceOp(op, catInputs[0]);
      return success();
    }

    // auto outputValWidth = hw::getBitWidth(op.getResult().getType());

    // remove nested
    bool anyInputDefinedByConcat = false;
    for (auto val: catInputs) {
      auto valDefiningOp = val.getDefiningOp();
      if (valDefiningOp == nullptr) return failure();
      if (isa<comb::ConcatOp>(valDefiningOp)) {
        anyInputDefinedByConcat = true;
      }
    }
    if (anyInputDefinedByConcat) {
      return handleNestedConcat(rewriter, op);
    }

    // Align
    if (!isElementsFullWidth(catInputs)) {
      if (!all_of(catInputs, [&](Value val){return hw::getBitWidth(val.getType()) <= 4;})) return failure();

      return alignInputs(rewriter, op);
    }

    return failure();
  }
};


struct LowerCombExtractOp: OpRewritePattern<comb::ExtractOp>, BitsOperations {
  using OpRewritePattern<comb::ExtractOp>::OpRewritePattern;

  LogicalResult handleNestedExtract(PatternRewriter &rewriter, comb::ExtractOp &extractOp, comb::ExtractOp &currentExtractOp, Value constZeroVal) const {
    auto lowBit = currentExtractOp.getLowBit();
    auto loc = currentExtractOp->getLoc();
    auto outputValWidth = hw::getBitWidth(currentExtractOp.getResult().getType());

    auto nestExtOpLowBit = extractOp.getLowBit();
    auto nestExtOpInputVal = extractOp.getInput();

    auto newLowBit = nestExtOpLowBit + lowBit;
    auto newExtractOp = rewriter.create<comb::ExtractOp>(loc, nestExtOpInputVal, newLowBit, outputValWidth);

    rewriter.replaceOp(currentExtractOp, newExtractOp);
    return success();
  }

  LogicalResult handleRegularConcatWithSingleOutput(PatternRewriter &rewriter, comb::ConcatOp &concatOp, comb::ExtractOp &currentExtractOp, Value constZeroVal) const {
    auto outputValWidth = hw::getBitWidth(currentExtractOp.getResult().getType());
    auto lowBit = currentExtractOp.getLowBit();
    auto loc = currentExtractOp->getLoc();

    // Require full aligned inputs
    auto catInputs = concatOp.getInputs();

    if (((lowBit / 4) != ((lowBit + outputValWidth - 1) / 4))) {
      // Input cross 4B boundary!
      size_t lowValPos = catInputs.size() - 1 - lowBit / 4;
      size_t lowValStart = lowBit % 4;
      size_t highValPos = catInputs.size() - 1 - (lowBit + outputValWidth) / 4;
      size_t highValEnd = (lowBit + outputValWidth) % 4;

      auto lowVal = bitsShrByNAndFill(rewriter, loc, catInputs[lowValPos], constZeroVal, lowValStart);
      auto highVal_mask = removeHighBits(rewriter, loc, catInputs[highValPos], highValEnd);
      auto highVal = bitsShlByNAndFill(rewriter, loc, highVal_mask, constZeroVal, 4 - lowValStart);

      auto resultType = rewriter.getIntegerType(outputValWidth);
      auto resultOp = rewriter.create<LUTOp>(loc, resultType, LUTOpName::LUT_Or, ValueRange({lowVal, highVal}));

      if (!hasSignalFragmentId(resultOp)) {
        copyCustomizedAttrs(currentExtractOp, resultOp);
      }
      assert(hw::getBitWidth(resultOp.getResult().getType()) == static_cast<int64_t>(outputValWidth));
      rewriter.replaceOp(currentExtractOp, resultOp);

      return success();
    } else {
      // Input within 4b boundary

      auto valPos = catInputs.size() - 1 - lowBit / 4;
      auto valStart = lowBit % 4;

      auto inputVal = catInputs[valPos];
      auto inputValWidth = hw::getBitWidth(inputVal.getType());
      if (valPos != 0) {
        assert(inputValWidth == 4);
      }

      auto val1 = bitsShrByNAndFill(rewriter, loc, inputVal, constZeroVal, valStart);
      auto val2 = removeHighBits(rewriter, loc, val1, outputValWidth);

      auto resultDefiningOp = val2.getDefiningOp();
      if (!hasSignalFragmentId(resultDefiningOp)) {
        copyCustomizedAttrs(currentExtractOp, resultDefiningOp);
      }

      rewriter.replaceOp(currentExtractOp, resultDefiningOp);
      return success();
    }
  }

  LogicalResult handleRegularConcatWithMultipleOutputSections(PatternRewriter &rewriter, comb::ConcatOp &concatOp, comb::ExtractOp &currentExtractOp, Value constZeroVal) const {
    auto outputValWidth = hw::getBitWidth(currentExtractOp.getResult().getType());
    auto lowBit = currentExtractOp.getLowBit();
    auto loc = currentExtractOp->getLoc();

    // Require full aligned inputs
    auto catInputs = concatOp.getInputs();

    if (!isElementsFullWidth(catInputs) || catInputs.size() == 1) {
      return failure();
    }
    SmallVector<Value> intermediateResults;
    size_t outpuChunks = (outputValWidth + 3) / 4;
    size_t shamt = lowBit % 4;
    size_t startingPos = (lowBit) / 4;
    size_t expectedLastChunkWidth = (outputValWidth % 4 == 0) ? 4 : outputValWidth % 4;

    for (size_t i = 0; i < outpuChunks; i++) {
      size_t pos = catInputs.size() - (startingPos + i + 1);
      if (shamt == 0) {
        intermediateResults.push_back(catInputs[pos]);
      } else {
        auto rhs = catInputs[pos];
        auto lhs = (pos != 0) ? catInputs[pos-1] : constZeroVal;

        auto shiftResult = bitsShrByN(rewriter, loc, lhs, rhs, shamt);

        intermediateResults.push_back(shiftResult);
      }
    }

    if (expectedLastChunkWidth < 4) {
      auto lastIndex = intermediateResults.size() - 1;
      auto limitWidthResult = removeHighBits(rewriter, loc, intermediateResults[lastIndex], expectedLastChunkWidth);
      intermediateResults[lastIndex] = limitWidthResult;
    }

    std::reverse(intermediateResults.begin(), intermediateResults.end());

    assert(isElementsFullWidth(intermediateResults) && "It should generate full width outputs");

    auto bitConcatOp = rewriter.create<comb::ConcatOp>(loc, intermediateResults);
    copyCustomizedAttrs(currentExtractOp, bitConcatOp);
    rewriter.replaceOp(currentExtractOp, bitConcatOp);

    return success();

  }

  LogicalResult handleNormalExtract(PatternRewriter &rewriter, comb::ExtractOp &currentExtractOp, Value constZeroVal) const {
    auto lowBit = currentExtractOp.getLowBit();
    auto loc = currentExtractOp->getLoc();
    auto outputValWidth = hw::getBitWidth(currentExtractOp.getResult().getType());

    auto inputVal = currentExtractOp.getInput();
    auto inputValWidth = hw::getBitWidth(inputVal.getType());


    auto valAlignedRight = extractHighNBitsAndShiftToLSB(rewriter, loc, inputVal, constZeroVal, inputValWidth - lowBit);
    auto valMaskTop = removeHighBits(rewriter, loc, valAlignedRight, outputValWidth);

    auto valDefiningOp = valMaskTop.getDefiningOp();
    if (!hasSignalFragmentId(valDefiningOp)) {
      copyCustomizedAttrs(currentExtractOp, valDefiningOp);
    }
    assert(hw::getBitWidth(valMaskTop.getType()) == static_cast<int64_t>(outputValWidth));
    rewriter.replaceOp(currentExtractOp, valMaskTop);

    return success();
  }

  LogicalResult matchAndRewrite(comb::ExtractOp op, PatternRewriter &rewriter) const final {
    // Remove unused
    if (op->getUsers().empty()) {
      // No body using. Remove
      rewriter.eraseOp(op);
      return success();
    }

    auto inputVal = op.getInput();
    auto outputVal = op.getResult();
    size_t inputValWidth = static_cast<size_t>(hw::getBitWidth(inputVal.getType()));
    size_t outputValWidth = static_cast<size_t>(hw::getBitWidth(outputVal.getType()));

    // auto lowBit = op.getLowBit();

    auto loc = op.getLoc();
    auto constZeroOp = rewriter.create<hw::ConstantOp>(loc, rewriter.getI4Type(), 0);
    auto constZeroVal = constZeroOp.getResult();

    auto inputDefiningOp = inputVal.getDefiningOp();
    if (inputDefiningOp == nullptr) return failure();

    if (auto extractOp = dyn_cast<comb::ExtractOp>(inputDefiningOp)) {
      // Nested extract
      // Dont' handle nested extract with multiple output 4b sections. they should be handled by previous canonicalizer.
      if (outputValWidth > 4) return failure();

      return handleNestedExtract(rewriter, extractOp, op, constZeroVal);

    } else if (auto concatOp = dyn_cast<comb::ConcatOp>(inputDefiningOp)) {
      // input is defined by concat
      auto catInputs = concatOp.getInputs();

      // Require full aligned inputs
      if (!isElementsFullWidth(catInputs) || catInputs.size() == 1) {
        return failure();
      }

      if (outputValWidth > 4) {
        return handleRegularConcatWithMultipleOutputSections(rewriter, concatOp, op, constZeroVal);
      } else {
        // Within 4B
        return handleRegularConcatWithSingleOutput(rewriter, concatOp, op, constZeroVal);
      }
    } else {
      // Not defined by concat
      if (inputValWidth > 4) return failure();
      return handleNormalExtract(rewriter, op, constZeroVal);
    }

    return failure();
  }
};

struct RemoveSeqToClockOp: OpRewritePattern<seq::ToClockOp> {
  using OpRewritePattern<seq::ToClockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(seq::ToClockOp op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerHWConstantOp: OpRewritePattern<hw::ConstantOp> {
  using OpRewritePattern<hw::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hw::ConstantOp op, PatternRewriter &rewriter) const final {
    SmallVector<Value> results;

    auto constValueWidth = op.getValue().getBitWidth();
    // auto constValueRaw = op.getValue().extractBits(0, 2);

    if (constValueWidth > 4) {
      auto chunks = split_signal_4B(constValueWidth);
      for (auto [chunkId, chunkWidth]: chunks) {
        auto newValue = op.getValue().extractBits(chunkWidth, chunkId * 4);

        auto newConstOp = rewriter.create<toucan::ConstantOp>(op->getLoc(), newValue);
        results.push_back(newConstOp.getResult());
      }

      auto bitConcatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), results);
      copyCustomizedAttrs(op, bitConcatOp);
      rewriter.replaceOp(op, bitConcatOp);
    } else {
      auto newConstOp = rewriter.create<toucan::ConstantOp>(op.getLoc(), op.getValue());
      copyCustomizedAttrs(op, newConstOp);
      rewriter.replaceOp(op, newConstOp);
    }
    return success();
  }
};




struct FactorConcatExtractPass : toucan::impl::FactorConcatExtractBase<FactorConcatExtractPass> {
  using FactorConcatExtractBase<FactorConcatExtractPass>::FactorConcatExtractBase;


  std::shared_ptr<FrozenRewritePatternSet> patterns;
  std::shared_ptr<ConversionTarget> target;

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    ConversionTarget conversionTarget(*context);
    
    owningPatterns.add<LowerCombConcatOp>(context);
    owningPatterns.add<LowerCombExtractOp>(context);
    owningPatterns.add<RemoveSeqToClockOp>(context);
    owningPatterns.add<LowerHWConstantOp>(context);

    // After conversion, only toucan and hw::Constant should exists
    conversionTarget.addLegalDialect<toucan::ToucanDialect>();
    conversionTarget.addLegalDialect<hw::HWDialect>();

    // After lowering, following ops should no longer appear
    conversionTarget.addIllegalOp<hw::HWModuleOp>();
    conversionTarget.addIllegalOp<hw::HWModuleExternOp>();
    conversionTarget.addIllegalOp<hw::ArrayGetOp>();

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    target = std::make_shared<ConversionTarget>(std::move(
    conversionTarget));

    return success();
  }



  LogicalResult runOnModule(hw::HWModuleOp mod) {
    auto converged = applyPatternsAndFoldGreedily(mod, *patterns);

    if (succeeded(converged)) return success();
    return success();
  }

  void parallelRunOnModules(ModuleOp &mod) {

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
  }



  void runOnOperation() final {
    auto mod = getOperation();

    if (parallel) {
      parallelRunOnModules(mod);
    } else {
      // Non parallel. do one by one
      auto ret = applyPatternsAndFoldGreedily(mod, *patterns);
      if (succeeded(ret))
        ;
    }
  }
};

std::unique_ptr<mlir::Pass> toucan::createFactorConcatExtractPass(FactorConcatExtractOptions options) {
  return std::make_unique<FactorConcatExtractPass>(options);
}
