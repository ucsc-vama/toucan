
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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <atomic>


#define GEN_PASS_DEF_LOWERCOMBTO4B_1
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombTo4B_1Pass"

#define ICMP_EQ_LENGTH_USE_VEC 1

static std::atomic<uint64_t> numLongRepInModule;
static std::atomic<uint64_t> numShortRepInModule;

static std::atomic<uint64_t> numCombShlInModules;
static std::atomic<uint64_t> numCombShrUInModules;
static std::atomic<uint64_t> numCombShrU1bInModule;
static std::atomic<uint64_t> numCombShrSInModules;
static std::atomic<uint64_t> numCombICmpInModules;
static std::atomic<uint64_t> numCombMulInModules;
static std::atomic<uint64_t> numCombParityInModules;
static std::atomic<uint64_t> numShiftToArrayInModules;
static std::atomic<uint64_t> numArrayReadFromShiftInModules;


struct LowerCombReplicateOp: OpRewritePattern<comb::ReplicateOp> {
  using OpRewritePattern<comb::ReplicateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ReplicateOp op, PatternRewriter &rewriter) const final {
    // Handles repop with input width > 1
    auto inputValue = op.getInput();
    auto inputValueWidth = hw::getBitWidth(inputValue.getType());
    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    if (inputValueWidth == 1) {
      numShortRepInModule++;

      assert(resultValueWidth > 1 && "Why you have a replicateOp with same input and output width?");
      
      if (resultValueWidth > 4) {
        SmallVector<Value> intermediateResults;
        for (auto&& [sigId, sigWidth]: split_signal_4B(resultValueWidth)) {
          if (sigWidth > 1) {
            auto newOp = rewriter.create<toucan::LUTOp>(op.getLoc(), LUTOpName::LUT_Rep1b, inputValue);
            auto repVal = newOp.getResult();
            // shink if needed
            auto shrinkVal = removeHighBits(rewriter, op.getLoc(), repVal, sigWidth);
            intermediateResults.push_back(shrinkVal);
          } else {
            // 1b. simply use input
            intermediateResults.push_back(inputValue);
          }
        }
        attachNameHintAndFragmentId(rewriter, intermediateResults, getSVNameHintAttr(op.getOperation()));

        concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
      } else {
        auto newOp = rewriter.create<toucan::LUTOp>(op.getLoc(), LUTOpName::LUT_Rep1b, inputValue);
        auto repVal = newOp.getResult();
        auto shrinkVal = removeHighBits(rewriter, op.getLoc(), repVal, resultValueWidth);

        attachNameHintAndFragmentId(rewriter, shrinkVal.getDefiningOp(), getSVNameHintAttr(newOp));

        rewriter.replaceOp(op, shrinkVal);
      }
    } else {
      // long
      numLongRepInModule++;

      auto resultValue = op.getResult();
      auto resultValueWidth = hw::getBitWidth(resultValue.getType());
      assert(resultValueWidth % inputValueWidth == 0);
      assert(resultValueWidth > inputValueWidth);

      auto numOf4B = static_cast<size_t>(resultValueWidth / inputValueWidth);

      SmallVector<Value> values;
      for (size_t i = 0; i < numOf4B; i++) {
        values.push_back(inputValue);
      }

      auto newOp = rewriter.create<comb::ConcatOp>(op.getLoc(), values);
      rewriter.replaceOp(op, newOp);
    }

    return success();
  }
};



class DynamicShiftOperations {
  public:

  // A vector-based shift right, use high bits of shamt
  // This logic is shared by shift left and right
  Value shiftCore(RewriterBase &rewriter, Operation *op, SmallVector<Value> intermediateResults, Value shamt, Value fillingValue, size_t realInputWidth, std::optional<StringAttr> namehint, bool isShiftLeft) const {
    auto shamtWidth = hw::getBitWidth(shamt.getType());

    SmallVector<Value> shiftResult;
    shiftResult.reserve(intermediateResults.size());

    if (shamtWidth > 2) {
      if (isShiftLeft) {
        std::reverse(intermediateResults.begin(), intermediateResults.end());
      }
      
      auto createVecOp = rewriter.create<toucan::DefVectorOp>(op->getLoc(), intermediateResults);
      auto vecHandle = createVecOp.getHandle();

      auto extractHighBitsOp = rewriter.create<comb::ExtractOp>(op->getLoc(), shamt, 2, shamtWidth - 2);
      auto shamt_high = extractHighBitsOp.getResult();

      auto shamt_high_split = (hw::getBitWidth(shamt_high.getType()) > 4) ? split_value_4B(op, shamt_high, rewriter) : SmallVector<Value>({shamt_high});
      
      for (size_t i = 0; i < intermediateResults.size(); i++) {
        uint16_t offset = intermediateResults.size() - 1 - i;
        auto vecReadOp = rewriter.create<toucan::VectorReadOp>(op->getLoc(), vecHandle, offset, fillingValue, shamt_high_split);
        shiftResult.push_back(vecReadOp.getResult());
      }
      if (isShiftLeft) {
        std::reverse(shiftResult.begin(), shiftResult.end());
      }
      numShiftToArrayInModules++;
      numArrayReadFromShiftInModules += shiftResult.size();
      
    } else {
      shiftResult = std::move(intermediateResults);
    }

    auto resultWidth = shiftResult.size() * 4;

    if (realInputWidth < resultWidth) {
      // extract leading bits
      auto extractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), shiftResult[0], 0, realInputWidth % 4);
      shiftResult[0] = extractOp.getResult();
    }

    if (shiftResult.size() > 1) {
      attachNameHintAndFragmentId(rewriter, shiftResult, namehint);
      auto concatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), shiftResult);
      return concatOp.getResult();
    } else {
      attachNameHintAndFragmentId(rewriter, shiftResult[0], namehint);
      return shiftResult[0];
    }
  }

  // shift left, inputs are 4b values from msb to lsb
  Value dshlCore(SmallVector<Value> &inputs, Value shamt, RewriterBase &rewriter, Operation* op, size_t realInputWidth, std::optional<StringAttr> namehint) const {

    auto shamtWidth = hw::getBitWidth(shamt.getType());

    auto int4BType = rewriter.getI4Type();
    auto zeroConst = rewriter.create<hw::ConstantOp>(op->getLoc(), int4BType, 0);
    auto zeroConstValue = zeroConst.getResult();

    uint64_t resultWidth = inputs.size() * 4;
    for (auto each_section: inputs) {
        auto sectionBitWidth = hw::getBitWidth(each_section.getType());
        assert(sectionBitWidth == 4);
    }

    auto result_sections = (resultWidth + 3) / 4;
    auto input_sections = inputs.size();
    assert(result_sections == input_sections);

    // intermediateResult[m][n] is inputs[m] << n
    SmallVector<Value> intermediateResults;
    intermediateResults.reserve(inputs.size());

    Value shamt_l2b;
    if (shamtWidth > 2) {
      auto extractLower2BOp = rewriter.create<comb::ExtractOp>(op->getLoc(), shamt, 0, 2);
      shamt_l2b = extractLower2BOp.getResult();
    } else {
      shamt_l2b = shamt;
    }

    for (size_t i = 0; i < inputs.size(); i++) {
      // Shift by 1, 2, and 3
      auto op1 = inputs[i];
      auto op2 = (i < inputs.size() - 1) ? inputs[i+1] : zeroConstValue;

      auto dshlOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_DShl, shamt_l2b, op1, op2);

      intermediateResults.push_back(dshlOp.getResult());
    }

    return shiftCore(rewriter, op, intermediateResults, shamt, zeroConstValue, realInputWidth, namehint, true);
  }

  // shift right, inputs need to be extended!!!, inputs are 4b values from msb to lsb
  Value dshrCore(SmallVector<Value> &inputs, Value shamt, RewriterBase &rewriter, Operation* op, size_t realInputWidth, Value fillingValue, std::optional<StringAttr> namehint) const {

    auto shamtWidth = hw::getBitWidth(shamt.getType());

    uint64_t resultWidth = inputs.size() * 4;
    for (auto each_section: inputs) {
        auto sectionBitWidth = hw::getBitWidth(each_section.getType());
        assert(sectionBitWidth == 4);
    }

    auto result_sections = (resultWidth + 3) / 4;
    auto input_sections = inputs.size();
    assert(result_sections == input_sections);

    // intermediateResult[m][n] is inputs[m] << n
    SmallVector<Value> intermediateResults;
    intermediateResults.reserve(inputs.size());

    Value shamt_l2b;
    if (shamtWidth > 2) {
      auto extractLower2BOp = rewriter.create<comb::ExtractOp>(op->getLoc(), shamt, 0, 2);
      shamt_l2b = extractLower2BOp.getResult();
    } else {
      shamt_l2b = shamt;
    }

    for (size_t i = 0; i < inputs.size(); i++) {
      // Shift by 1, 2, and 3
      auto op1 = (i == 0) ? fillingValue : inputs[i-1];
      auto op2 = inputs[i];

      auto dshrOp = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_DShr, shamt_l2b, op1, op2);

      intermediateResults.push_back(dshrOp.getResult());
    }

    return shiftCore(rewriter, op, intermediateResults, shamt, fillingValue, realInputWidth, namehint, false);
  }



  // This function may modify shamtValue!
  // For now, don't check if shamt has extra bits. leave it to canonicalizer.
  size_t limitShamtWidth(Value &inputValue, Value &shamtValue, RewriterBase &rewriter, Operation *op) const {
    auto inputValueWidth = hw::getBitWidth(inputValue.getType());
    auto necessaryShamtBits = llvm::Log2_64_Ceil(inputValueWidth);
    auto shamtWidth = hw::getBitWidth(shamtValue.getType());

    if (shamtWidth > necessaryShamtBits) {
      op->emitWarning() << "Shamt too large: " << necessaryShamtBits << " bits of shamt is sufficient for " << inputValueWidth << " width inputs, but got " << shamtWidth << ". Extra bits will be trimmed.";
      auto shamtExtractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), shamtValue, 0, necessaryShamtBits);
      shamtValue = shamtExtractOp.getResult();
      shamtWidth = necessaryShamtBits;
    }
    return shamtWidth;
  }
};


struct LowerCombShlOp: OpRewritePattern<comb::ShlOp>, DynamicShiftOperations {
  using OpRewritePattern<comb::ShlOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ShlOp shlOp, PatternRewriter &rewriter) const final {
    auto inputValue = shlOp.getLhs();
    auto shamtValue = extractMinimumWidth(shlOp.getRhs(), rewriter, shlOp.getOperation());
    auto shamtWidth = hw::getBitWidth(shamtValue.getType());
    if (shamtWidth > 30) {
      shlOp.emitError("shl: Shift amount is too large: " + std::to_string(shamtWidth));
      return failure();
    }
    numCombShlInModules++;

    // shamtWidth = limitShamtWidth(inputValue, shamtValue, rewriter, shlOp.getOperation());
    assert(hw::getBitWidth(shamtValue.getType()) == shamtWidth);

    // Split input signals

    auto inputValueWithPadding = padding_with_0_and_align_4b(shlOp.getOperation(), rewriter, inputValue);
    auto inputValuesWithPadding_4b = split_value_4B(shlOp.getOperation(), inputValueWithPadding, rewriter);

    auto optionalNameHint = getSVNameHintAttr(shlOp);

    auto result = dshlCore(inputValuesWithPadding_4b, shamtValue, rewriter, shlOp.getOperation(), hw::getBitWidth(inputValue.getType()), optionalNameHint);

    auto oldResult = shlOp.getResult();
    auto oldResultWidth = hw::getBitWidth(oldResult.getType());
    auto newResultWidth = hw::getBitWidth(result.getType());
    assert(oldResultWidth == newResultWidth);

    rewriter.replaceOp(shlOp, result);

    return success();
  }
};

struct LowerCombShrUOp: OpRewritePattern<comb::ShrUOp>, DynamicShiftOperations {
  using OpRewritePattern<comb::ShrUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ShrUOp shruOp, PatternRewriter &rewriter) const final {
    numCombShrUInModules++;

    auto inputValue = shruOp.getLhs();
    auto shamtValue = extractMinimumWidth(shruOp.getRhs(), rewriter, shruOp.getOperation());
    auto shamtWidth = hw::getBitWidth(shamtValue.getType());
    if (shamtWidth > 30) {
      shruOp.emitError("shru: Shift amount is too large: " + std::to_string(shamtWidth));
      return failure();
    }

    auto resultWidth = hw::getBitWidth(shruOp.getResult().getType());
    if (resultWidth == 1) {
      // 1 bit. Replace with mux.
      numCombShrU1bInModule++;

      auto en = shruOp.getRhs();
      auto data = shruOp.getLhs();
      assert(hw::getBitWidth(en.getType()) == 1);
      assert(hw::getBitWidth(data.getType()) == 1);

      auto constZeroOp = rewriter.create<hw::ConstantOp>(shruOp.getLoc(), rewriter.getI1Type(), 0);
      auto constZeroValue = constZeroOp.getResult();

      auto muxOp = rewriter.create<comb::MuxOp>(shruOp.getLoc(), en, constZeroValue, data);

      rewriter.replaceOp(shruOp, muxOp);
      return success();
    }


    auto inputValueWithPadding = padding_with_0_and_align_4b(shruOp.getOperation(), rewriter, inputValue);
    auto inputValuesWithPadding_4b = split_value_4B(shruOp.getOperation(), inputValueWithPadding, rewriter);

    auto inputValueWidth = hw::getBitWidth(inputValue.getType());
    assert(inputValueWidth > 1);


    auto zeroConstOp = rewriter.create<hw::ConstantOp>(shruOp.getLoc(), rewriter.getI4Type(), 0);
    auto zeroConstValue = zeroConstOp.getResult();

    auto optionalNameHint = getSVNameHintAttr(shruOp);

    auto result = dshrCore(inputValuesWithPadding_4b, shamtValue, rewriter, shruOp.getOperation(), inputValueWidth, zeroConstValue, optionalNameHint);

    auto oldResult = shruOp.getResult();
    auto oldResultWidth = hw::getBitWidth(oldResult.getType());
    auto newResultWidth = hw::getBitWidth(result.getType());
    assert(oldResultWidth == newResultWidth);

    rewriter.replaceOp(shruOp, result);

    return success();
  }
};


struct LowerCombShrSOp: OpRewritePattern<comb::ShrSOp>, DynamicShiftOperations {
  using OpRewritePattern<comb::ShrSOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ShrSOp shrsOp, PatternRewriter &rewriter) const final {
    auto inputValue = shrsOp.getLhs();
    auto shamtValue = extractMinimumWidth(shrsOp.getRhs(), rewriter, shrsOp.getOperation());
    auto shamtWidth = hw::getBitWidth(shamtValue.getType());
    if (shamtWidth > 30) {
      shrsOp.emitError("shrs: Shift amount is too large: " + std::to_string(shamtWidth));
      return failure();
    }
    numCombShrSInModules++;

    auto inputValueWidth = hw::getBitWidth(inputValue.getType());
    assert(inputValueWidth > 1);

    // shamtWidth = limitShamtWidth(inputValue, shamtValue, rewriter, shrsOp.getOperation());
    assert(hw::getBitWidth(shamtValue.getType()) == shamtWidth);

    auto zeroConst4bOp = rewriter.create<hw::ConstantOp>(shrsOp.getLoc(), rewriter.getI4Type(), 0);
    auto zeroConst4bValue = zeroConst4bOp.getResult();
    auto oneConst4bOp = rewriter.create<hw::ConstantOp>(shrsOp.getLoc(), rewriter.getI4Type(), 0xF);
    auto oneConst4bValue = oneConst4bOp.getResult();

    // Split input signals
    auto inputValues_4b = split_value_4B(shrsOp.getOperation(), inputValue, rewriter);
    auto headFragmentWidth = hw::getBitWidth(inputValues_4b[0].getType());

    auto msbExtactOp = rewriter.create<comb::ExtractOp>(shrsOp.getLoc(), inputValues_4b[0], headFragmentWidth - 1, 1);
    auto isNeg = msbExtactOp.getResult();

    // calculate filling value
    auto fillingMuxOp = rewriter.create<comb::MuxOp>(shrsOp.getLoc(), isNeg, oneConst4bValue, zeroConst4bValue);
    auto fillingValue = fillingMuxOp.getResult();

    if (headFragmentWidth != 4) {
      assert(headFragmentWidth < 4);
      // calculate top bits
      auto paddingBits = 4 - headFragmentWidth;
      auto fragmentFillingOp = rewriter.create<comb::ExtractOp>(shrsOp.getLoc(), fillingValue, 0, paddingBits);
      auto fragmentFillingValue = fragmentFillingOp.getResult();

      // Temp value, don't need namehint
      auto concatOp = rewriter.create<comb::ConcatOp>(shrsOp.getLoc(), fragmentFillingValue, inputValues_4b[0]);

      inputValues_4b[0] = concatOp.getResult();
    }

    auto optionalNameHint = getSVNameHintAttr(shrsOp);

    auto result = dshrCore(inputValues_4b, shamtValue, rewriter, shrsOp.getOperation(), inputValueWidth, fillingValue, optionalNameHint);

    auto oldResult = shrsOp.getResult();
    auto oldResultWidth = hw::getBitWidth(oldResult.getType());
    auto newResultWidth = hw::getBitWidth(result.getType());
    assert(oldResultWidth == newResultWidth);

    rewriter.replaceOp(shrsOp, result);

    return success();
  }
};


struct LowerCombICmpOp: OpRewritePattern<comb::ICmpOp> {
  using OpRewritePattern<comb::ICmpOp>::OpRewritePattern;

  comb::ICmpPredicate getImplementablePredicate(comb::ICmpPredicate predicate) const {
    switch (predicate) {
      case comb::ICmpPredicate::eq:
      case comb::ICmpPredicate::ne:
        return comb::ICmpPredicate::eq;
      case comb::ICmpPredicate::slt:
      case comb::ICmpPredicate::sge:
        return comb::ICmpPredicate::slt;
      case comb::ICmpPredicate::sle:
      case comb::ICmpPredicate::sgt:
        return comb::ICmpPredicate::sle;
      case comb::ICmpPredicate::ult:
      case comb::ICmpPredicate::uge:
        return comb::ICmpPredicate::ult;
      case comb::ICmpPredicate::ule:
      case comb::ICmpPredicate::ugt:
        return comb::ICmpPredicate::ule;
      case comb::ICmpPredicate::ceq:
      case comb::ICmpPredicate::cne:
      case comb::ICmpPredicate::weq:
      case comb::ICmpPredicate::wne:
        return comb::ICmpPredicate::ceq;
    }
    llvm_unreachable("Unknow icmp predicate!");
  }

  Value addNot(comb::ICmpOp &op, PatternRewriter &rewriter, Value &value) const {
    auto constOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getI1Type(), 1);
    auto constVal = constOp.getResult();
    auto notOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Xor, value, constVal);
    return notOp.getResult();
  }

  // Unlike padding_with_0_and_align_4b,
  // This step ensures add at least 1 bit (in fact adds 1~4 bits)
  // Ensures the leading bits is always 0
  // Thus convert an unsigned ICmp to signed.
  Value paddingValueWithZeroForIcmp(comb::ICmpOp &op, PatternRewriter &rewriter, Value val) const {
    auto inputValueWidth = hw::getBitWidth(val.getType());

    auto paddingTargetWidth = ((inputValueWidth & 0x3) == 0) ? inputValueWidth + 4 : ((inputValueWidth + 3) & (~0x3));

    auto constOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getIntegerType(paddingTargetWidth - inputValueWidth), 0);
    auto constVal = constOp.getResult();

    // Temp value, don't need namehint
    auto paddingOp = rewriter.create<comb::ConcatOp>(op.getLoc(), ValueRange({constVal, val}));
    auto paddingValue = paddingOp.getResult();

    return paddingValue;
  }


  // This procedure allows wide comparison
  Value icmpEqCore(comb::ICmpOp &op, PatternRewriter &rewriter, Value lhsValue, Value rhsValue) const {
    assert(hw::getBitWidth(lhsValue.getType()) == hw::getBitWidth(rhsValue.getType()));
    auto inputBitWidth = hw::getBitWidth(lhsValue.getType());

    if (inputBitWidth <= 4) {
      // only 1 element. In this case, simply use LUT
      auto cmpLutOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Cmp_Eq, lhsValue, rhsValue);

      return cmpLutOp.getResult();
    } else {
      // multiple elements. Use VecLogicOp
      auto lhsPadding = padding_with_0_and_align_4b(op, rewriter, lhsValue);
      auto rhsPadding = padding_with_0_and_align_4b(op, rewriter, rhsValue);

      auto allLhsValues = split_value_4B(op.getOperation(), lhsPadding, rewriter);
      auto allRhsValues = split_value_4B(op.getOperation(), rhsPadding, rewriter);

      mlir::SmallVector<mlir::Value> resultVals, lhsValues, rhsValues;

      int remainingSections = static_cast<int>(allLhsValues.size());
      while (remainingSections > 0) {
        int sectionsInThisIteration = std::min(remainingSections, TOUCAN_VEC_OP_MAX_SECTIONS);
        remainingSections -= sectionsInThisIteration;

        if (sectionsInThisIteration <= ICMP_EQ_LENGTH_USE_VEC) {
          // Too small to use vec op
          assert(allLhsValues.size() == static_cast<size_t>(sectionsInThisIteration));
          for (int i = 0; i < sectionsInThisIteration; i++) {
            auto cmpLutOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Cmp_Eq, allLhsValues[i], allRhsValues[i]);
            resultVals.push_back(cmpLutOp.getResult());
          }
        } else {
          // Use vec op
          lhsValues.clear();
          rhsValues.clear();

          assert(sectionsInThisIteration <= static_cast<int>(allLhsValues.size()));

          lhsValues.append(allLhsValues.begin(), allLhsValues.begin() + sectionsInThisIteration);
          rhsValues.append(allRhsValues.begin(), allRhsValues.begin() + sectionsInThisIteration);

          allLhsValues.erase(allLhsValues.begin(), allLhsValues.begin() + sectionsInThisIteration);
          allRhsValues.erase(allRhsValues.begin(), allRhsValues.begin() + sectionsInThisIteration);

          auto lhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), lhsValues);
          auto rhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), rhsValues);

          auto lhsVecHandle = lhsVecDeclOp.getHandle();
          auto rhsVecHandle = rhsVecDeclOp.getHandle();

          auto vecCmpEqOp = rewriter.create<toucan::VectorLogicOp>(op.getLoc(), toucan::VecLogicOpName::VecLogic_Eq, lhsVecHandle, rhsVecHandle);

          resultVals.push_back(vecCmpEqOp.getResult());
        }

      }

      assert(resultVals.size() != 0);

      if (resultVals.size() == 1) {
        return resultVals[0];
      } else {
        // Bitwise and of all them
        auto result = generate_reduce_tree(rewriter, op.getLoc(), resultVals, [&](RewriterBase &rewriter, Location loc, Value lhs, Value rhs) {
          auto andOp = rewriter.create<comb::AndOp>(loc, ValueRange({lhs, rhs}), false);
          return andOp.getResult();
        });
        return result;
      }
    }
  }

  Value icmpLTCore(comb::ICmpOp &op, PatternRewriter &rewriter, Value lhs, Value rhs, bool isULT) const {
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));
    auto inputBitWidth = hw::getBitWidth(lhs.getType());
    assert(inputBitWidth <= TOUCAN_VEC_OP_MAX_WIDTH);
    if (!isULT) {
      // signed. leave 1 bit
      assert(inputBitWidth != TOUCAN_VEC_OP_MAX_WIDTH);
    }

    if (inputBitWidth <= 4) {
      // only 1 element. In this case, simply use LUT
      if (!isULT) {

        if (inputBitWidth == 4) {
          auto sltOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Cmp_Slt4b, lhs, rhs);

          return sltOp.getResult();
        } else {
          auto subLutOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Sub, lhs, rhs);

          auto extractMSBOp = rewriter.create<comb::ExtractOp>(op.getLoc(), subLutOp.getResult(), 3, 1);

          return extractMSBOp.getResult();
        }
      } else {
        auto ultOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Cmp_Ult, lhs, rhs);

        return ultOp.getResult();
      }
    } else {
      // multiple elements. Use VecLogicOp
      assert(inputBitWidth % 4 == 0);

      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      if (lhsValues.size() > TOUCAN_VEC_OP_MAX_SECTIONS) {
        llvm::errs() << "LT operation is too wide, not supported yet!\n";
        llvm_unreachable("Comparison for int wider than 128 is not supported. Check icmpEqCore for reference implementation");
      }

      auto lhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), lhsValues);
      auto rhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), rhsValues);

      auto lhsVecHandle = lhsVecDeclOp.getHandle();
      auto rhsVecHandle = rhsVecDeclOp.getHandle();

      auto vecCmpOp = rewriter.create<toucan::VectorLogicOp>(op.getLoc(), toucan::VecLogicOpName::VecLogic_Lt, lhsVecHandle, rhsVecHandle);

      return vecCmpOp.getResult();
    }
  }

  Value icmpLECore(comb::ICmpOp &op, PatternRewriter &rewriter, Value lhs, Value rhs, bool isULE) const {
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));
    auto inputBitWidth = hw::getBitWidth(lhs.getType());

    assert(inputBitWidth <= TOUCAN_VEC_OP_MAX_WIDTH);
    if (!isULE) {
      // signed. leave 1 bit
      assert(inputBitWidth != TOUCAN_VEC_OP_MAX_WIDTH);
    }

    if (inputBitWidth <= 4) {
      // only 1 element. In this case, simply use LUT
      auto isLessThan = icmpLTCore(op, rewriter, lhs, rhs, isULE);

      auto cmpEqLutOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Cmp_Eq, lhs, rhs);
      auto isEqual = cmpEqLutOp.getResult();

      auto orOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Or, isLessThan, isEqual);

      return orOp.getResult();
    } else {
      // multiple elements. Use VecLogicOp
      // Note: add at least 1 extra bit
      assert(inputBitWidth % 4 == 0);

      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      if (lhsValues.size() > TOUCAN_VEC_OP_MAX_SECTIONS) {
        llvm::errs() << "LE operation is too wide, not supported yet!\n";
        llvm_unreachable("Comparison for int wider than 128 is not supported. Check icmpEqCore for reference implementation");
      }

      auto lhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), lhsValues);
      auto rhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), rhsValues);

      auto lhsVecHandle = lhsVecDeclOp.getHandle();
      auto rhsVecHandle = rhsVecDeclOp.getHandle();

      auto vecCmpOp = rewriter.create<toucan::VectorLogicOp>(op.getLoc(), toucan::VecLogicOpName::VecLogic_Le, lhsVecHandle, rhsVecHandle);

      return vecCmpOp.getResult();
    }
  }

  Value LowerCombICmpEQ(comb::ICmpOp &op, PatternRewriter &rewriter) const {
    auto lhsValue = op.getLhs();
    auto rhsValue = op.getRhs();

    auto inputValueWidth = hw::getBitWidth(lhsValue.getType());
    assert(inputValueWidth == hw::getBitWidth(rhsValue.getType()));

    auto result = icmpEqCore(op, rewriter, lhsValue, rhsValue);
    return result;
  }


  Value LowerCombICmpULT(comb::ICmpOp &op, PatternRewriter &rewriter) const {
    auto lhsValue = op.getLhs();
    auto rhsValue = op.getRhs();

    auto inputValueWidth = hw::getBitWidth(lhsValue.getType());
    assert(inputValueWidth == hw::getBitWidth(rhsValue.getType()));

    if (inputValueWidth <= 4) {
      return icmpLTCore(op, rewriter, lhsValue, rhsValue, true);
    } else {
      auto paddingLhsValue = paddingValueWithZeroForIcmp(op, rewriter, lhsValue);
      auto paddingRhsValue = paddingValueWithZeroForIcmp(op, rewriter, rhsValue);

      return icmpLTCore(op, rewriter, paddingLhsValue, paddingRhsValue, true);
    }
  }

  Value LowerCombICmpULE(comb::ICmpOp &op, PatternRewriter &rewriter) const {
    auto lhsValue = op.getLhs();
    auto rhsValue = op.getRhs();

    auto inputValueWidth = hw::getBitWidth(lhsValue.getType());
    assert(inputValueWidth == hw::getBitWidth(rhsValue.getType()));

    if (inputValueWidth <= 4) {
      return icmpLECore(op, rewriter, lhsValue, rhsValue, true);
    } else {
      auto paddingLhsValue = paddingValueWithZeroForIcmp(op, rewriter, lhsValue);
      auto paddingRhsValue = paddingValueWithZeroForIcmp(op, rewriter, rhsValue);

      return icmpLECore(op, rewriter, paddingLhsValue, paddingRhsValue, true);
    }
  }

  Value LowerCombICmpSLT(comb::ICmpOp &op, PatternRewriter &rewriter) const {
    auto lhsValue = op.getLhs();
    auto rhsValue = op.getRhs();

    auto inputValueWidth = hw::getBitWidth(lhsValue.getType());
    assert(inputValueWidth == hw::getBitWidth(rhsValue.getType()));

    if (inputValueWidth <= 4) {
      auto sExtLhsValue = signExt_4b(rewriter, op.getLoc(), lhsValue);
      auto sExtRhsValue = signExt_4b(rewriter, op.getLoc(), rhsValue);

      return icmpLTCore(op, rewriter, sExtLhsValue, sExtRhsValue, false);
    } else {
      auto sExtLhsValue = signExtValueToNext4b(rewriter, op.getLoc(), lhsValue);
      auto sExtRhsValue = signExtValueToNext4b(rewriter, op.getLoc(), rhsValue);

      return icmpLTCore(op, rewriter, sExtLhsValue, sExtRhsValue, false);
    }
  }

  Value LowerCombICmpSLE(comb::ICmpOp &op, PatternRewriter &rewriter) const {
    auto lhsValue = op.getLhs();
    auto rhsValue = op.getRhs();

    auto inputValueWidth = hw::getBitWidth(lhsValue.getType());
    assert(inputValueWidth == hw::getBitWidth(rhsValue.getType()));

    if (inputValueWidth <= 4) {
      auto sExtLhsValue = signExt_4b(rewriter, op.getLoc(), lhsValue);
      auto sExtRhsValue = signExt_4b(rewriter, op.getLoc(), rhsValue);

      return icmpLECore(op, rewriter, sExtLhsValue, sExtRhsValue, false);
    } else {
      auto sExtLhsValue = signExtValueToNext4b(rewriter, op.getLoc(), lhsValue);
      auto sExtRhsValue = signExtValueToNext4b(rewriter, op.getLoc(), rhsValue);

      return icmpLECore(op, rewriter, sExtLhsValue, sExtRhsValue, false);
    }
  }

  LogicalResult matchAndRewrite(comb::ICmpOp op, PatternRewriter &rewriter) const final {
    numCombICmpInModules++;

    auto predicate = op.getPredicate();
    auto implPredicate = getImplementablePredicate(predicate);

    Value result;

    switch (implPredicate) {
      case circt::comb::ICmpPredicate::eq : {
        result = LowerCombICmpEQ(op, rewriter);
        break;
      }
      case circt::comb::ICmpPredicate::slt: {
        result = LowerCombICmpSLT(op, rewriter);
        break;
      }
      case circt::comb::ICmpPredicate::sle: {
        result = LowerCombICmpSLE(op, rewriter);
        break;
      }
      case circt::comb::ICmpPredicate::ult: {
        result = LowerCombICmpULT(op, rewriter);
        break;
      }
      case circt::comb::ICmpPredicate::ule: {
        result = LowerCombICmpULE(op, rewriter);
        break;
      }
      case circt::comb::ICmpPredicate::ceq: {
        // not supported
        op.emitError() << "ceq, cne, weq, wne are not supported";
        return failure();
      }
      default: {
        llvm_unreachable("Should not reach here");
        return failure();
      }
    }

    if (predicate != implPredicate) {
      // Add a not gate
      result = addNot(op, rewriter, result);
    }

    auto optionalNameHint = getSVNameHintAttr(op);
    attachNameHintAndFragmentId(rewriter, result, optionalNameHint);
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct LowerCombMulOp: OpRewritePattern<comb::MulOp> {
  using OpRewritePattern<comb::MulOp>::OpRewritePattern;

  const size_t maxMulWidth = 128;

  LogicalResult matchAndRewrite(comb::MulOp op, PatternRewriter &rewriter) const final {
    auto inputs = op.getInputs();
    assert(inputs.size() == 2);
    auto lhsValue = inputs[0];
    auto rhsValue = inputs[1];

    assert(hw::getBitWidth(lhsValue.getType()) == hw::getBitWidth(rhsValue.getType()));
    assert(hw::getBitWidth(lhsValue.getType()) <= TOUCAN_VEC_OP_MAX_WIDTH);

    auto inputBitWidth = hw::getBitWidth(lhsValue.getType());
    if (static_cast<size_t>(inputBitWidth) > maxMulWidth) {
      // Mul wider than this value may be too costly. 
      op->emitError() << "Multiplication too wide (max limit is " << maxMulWidth << ", got " << inputBitWidth << ")";
      return failure();
    }
    numCombMulInModules++;

    auto lhsPadding = padding_with_0_and_align_4b(op.getOperation(), rewriter, lhsValue);
    auto rhsPadding = padding_with_0_and_align_4b(op.getOperation(), rewriter, rhsValue);

    auto lhsValues = split_value_4B(op.getOperation(), lhsPadding, rewriter);
    auto rhsValues = split_value_4B(op.getOperation(), rhsPadding, rewriter);

    auto lhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), lhsValues);
    auto rhsVecDeclOp = rewriter.create<toucan::DefVectorOp>(op.getLoc(), rhsValues);

    auto lhsVecHandle = lhsVecDeclOp.getHandle();
    auto rhsVecHandle = rhsVecDeclOp.getHandle();

    auto vecMulOp = rewriter.create<toucan::VectorArithOp>(op.getLoc(), toucan::VecArithOpName::VecArith_Mul, lhsVecHandle, rhsVecHandle);
    auto vecMulResultVec = vecMulOp.getResult();

    auto mulVal = convertFullVectorBackToValue(rewriter, op.getLoc(), vecMulResultVec);

    // Shrink to desired width
    auto shrinkOp = rewriter.create<comb::ExtractOp>(op.getLoc(), mulVal, 0, inputBitWidth);
    auto result = shrinkOp.getResult();

    copyCustomizedAttrs(op, result.getDefiningOp());
    rewriter.replaceOp(op, result);

    return success();
  }
};



struct LowerCombParityOp: OpRewritePattern<comb::ParityOp> {
  using OpRewritePattern<comb::ParityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ParityOp op, PatternRewriter &rewriter) const final {
    numCombParityInModules++;

    auto inputValue = op.getInput();
    auto inputValueWidth = hw::getBitWidth(inputValue.getType());

    if (inputValueWidth > 4) {
      SmallVector<Value> intermediateResults;
      for (auto splitVal: split_value_4B(op, inputValue, rewriter)) {
        auto xorrOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_XorR, splitVal);
        auto xorrResult = xorrOp.getResult();
        // assert(hw::getBitWidth(xorrResult.getType()) == 1);
        intermediateResults.push_back(xorrResult);
      }
      auto result = generate_reduce_tree(rewriter, op.getLoc(), intermediateResults, [&](RewriterBase &rewriter, Location loc, Value lhs, Value rhs) {
        auto xorOp = rewriter.create<comb::XorOp>(loc, ValueRange({lhs, rhs}), false);
        return xorOp.getResult();
      });

      assert(hw::getBitWidth(result.getType()) == 1);

      copyCustomizedAttrs(op, result.getDefiningOp());

      rewriter.replaceOp(op, result);
    } else {
      auto xorrOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_XorR, inputValue);


      copyCustomizedAttrs(op, xorrOp);
      rewriter.replaceOp(op, xorrOp);
    }

    return success();
  }
};


struct LowerCombDivSOp: OpRewritePattern<comb::DivSOp> {
  using OpRewritePattern<comb::DivSOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::DivSOp op, PatternRewriter &rewriter) const final {
    op->emitError("Operation not supported");
    return failure();
  }
};

struct LowerCombDivUOp: OpRewritePattern<comb::DivUOp> {
  using OpRewritePattern<comb::DivUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::DivUOp op, PatternRewriter &rewriter) const final {
    op->emitError("Operation not supported");
    return failure();
  }
};

struct LowerCombModSOp: OpRewritePattern<comb::ModSOp> {
  using OpRewritePattern<comb::ModSOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ModSOp op, PatternRewriter &rewriter) const final {
    op->emitError("Operation not supported");
    return failure();
  }
};

struct LowerCombModUOp: OpRewritePattern<comb::ModUOp> {
  using OpRewritePattern<comb::ModUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::ModUOp op, PatternRewriter &rewriter) const final {
    op->emitError("Operation not supported");
    return failure();
  }
};




struct LowerCombTo4B_1Pass : toucan::impl::LowerCombTo4B_1Base<LowerCombTo4B_1Pass> {
  using LowerCombTo4B_1Base<LowerCombTo4B_1Pass>::LowerCombTo4B_1Base;

  std::shared_ptr<FrozenRewritePatternSet> patterns;
  std::shared_ptr<ConversionTarget> target;

  LogicalResult initialize(MLIRContext *context) override {
    numLongRepInModule = 0;
    numShortRepInModule = 0;
    numCombShlInModules = 0;
    numCombShrUInModules = 0;
    numCombShrU1bInModule = 0;
    numCombShrSInModules = 0;
    numCombICmpInModules = 0;
    numCombMulInModules = 0;
    numCombParityInModules = 0;
    numShiftToArrayInModules = 0;
    numArrayReadFromShiftInModules = 0;

    RewritePatternSet owningPatterns(context);
    ConversionTarget conversionTarget(*context);
    
    owningPatterns.add<LowerCombReplicateOp>(context);
    owningPatterns.add<LowerCombShlOp>(context);
    owningPatterns.add<LowerCombShrUOp>(context);
    owningPatterns.add<LowerCombShrSOp>(context);
    owningPatterns.add<LowerCombICmpOp>(context);
    owningPatterns.add<LowerCombMulOp>(context);
    owningPatterns.add<LowerCombParityOp>(context);

    // Those operations are not supported yet
    owningPatterns.add<LowerCombDivUOp>(context);
    owningPatterns.add<LowerCombDivSOp>(context);
    owningPatterns.add<LowerCombModUOp>(context);
    owningPatterns.add<LowerCombModSOp>(context);

    conversionTarget.addLegalDialect<toucan::ToucanDialect>();
    conversionTarget.addLegalDialect<hw::HWDialect>();
    conversionTarget.addLegalDialect<comb::CombDialect>();

    // After lowering, following ops should no longer appear
    conversionTarget.addIllegalOp<comb::ReplicateOp>();
    conversionTarget.addIllegalOp<comb::ShlOp>();
    conversionTarget.addIllegalOp<comb::ShrUOp>();
    conversionTarget.addIllegalOp<comb::ShrSOp>();
    conversionTarget.addIllegalOp<comb::ICmpOp>();
    conversionTarget.addIllegalOp<comb::MulOp>();
    conversionTarget.addIllegalOp<comb::ParityOp>();

    conversionTarget.addIllegalOp<comb::DivUOp>();
    conversionTarget.addIllegalOp<comb::DivSOp>();
    conversionTarget.addIllegalOp<comb::ModUOp>();
    conversionTarget.addIllegalOp<comb::ModSOp>();

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    target = std::make_shared<ConversionTarget>(std::move(
    conversionTarget));

    return success();
  }


  LogicalResult runOnModule(hw::HWModuleOp mod) {
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

    numLongRep = numLongRepInModule;
    numShortRep = numShortRepInModule;
    numCombShl = numCombShlInModules;
    numCombShrU = numCombShrUInModules;
    numCombShrU1b = numCombShrU1bInModule;
    numCombShrS = numCombShrSInModules;
    numCombICmp = numCombICmpInModules;
    numCombMul = numCombMulInModules;
    numCombParity = numCombParityInModules;
    numShiftToArray = numShiftToArrayInModules;
    numArrayReadFromShift = numArrayReadFromShiftInModules;
  }
};

std::unique_ptr<mlir::Pass> toucan::createLowerCombTo4B_1Pass() {
  return std::make_unique<LowerCombTo4B_1Pass>();
}
