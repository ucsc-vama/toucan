
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
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <memory>
#include <string>


#define GEN_PASS_DEF_LOWERCOMBTO4B_1
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombTo4B_1Pass"

struct LowerCombTo4B_1Pass : toucan::impl::LowerCombTo4B_1Base<LowerCombTo4B_1Pass> {
  using LowerCombTo4B_1Base<LowerCombTo4B_1Pass>::LowerCombTo4B_1Base;

  LogicalResult splitHWConstOp(hw::ConstantOp &op) {
    SmallVector<Value> results;

    auto constValueWidth = op.getValue().getBitWidth();
    // auto constValueRaw = op.getValue().extractBits(0, 2);

    if (constValueWidth > 4) {
      OpBuilder builder(op);
      IRRewriter rewriter(builder);
      
      auto chunks = split_signal_4B(constValueWidth);
      for (auto [chunkId, chunkWidth]: chunks) {
        auto newValue = op.getValue().extractBits(chunkWidth, chunkId * 4);
        auto newConstOp = rewriter.create<hw::ConstantOp>(op->getLoc(), newValue);
        results.push_back(newConstOp.getResult());
      }

      auto bitConcatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), results);
      rewriter.replaceAllUsesWith(op, bitConcatOp);
      // consider remove op
    }
    return success();
  }

  LogicalResult splitCombAndOp(comb::AndOp &op) {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
      return failure();
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    OpBuilder builder(op);
    IRRewriter rewriter(builder);

    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_And, lhs, rhs);
        intermediateResults.push_back(andOp.getResult());
      }

      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_And, lhs, rhs);
      tryCopySVNameHint(op.getOperation(), andOp.getOperation());
      rewriter.replaceAllUsesWith(op, andOp);
    }
    
    return success();
  }


  

  LogicalResult splitCombOrOp(comb::OrOp &op) {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
      return failure();
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());
        
    OpBuilder builder(op);
    IRRewriter rewriter(builder);
    
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto orOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Or, lhs, rhs);
        intermediateResults.push_back(orOp.getResult());
      }

      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto orOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Or, lhs, rhs);
      tryCopySVNameHint(op.getOperation(), orOp.getOperation());
      rewriter.replaceAllUsesWith(op, orOp);
    }
    
    return success();
  }

  LogicalResult splitCombXorOp(comb::XorOp &op) {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
      return failure();
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    OpBuilder builder(op);
    IRRewriter rewriter(builder);
    
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto xorOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Xor, lhs, rhs);
        intermediateResults.push_back(xorOp.getResult());
      }

      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto xorOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Xor, lhs, rhs);
      tryCopySVNameHint(op.getOperation(), xorOp.getOperation());
      rewriter.replaceAllUsesWith(op, xorOp);
    }
    
    return success();
  }

  LogicalResult splitCombReplicate(comb::ReplicateOp &repOp) {
    auto inputValue = repOp.getInput();
    if (hw::getBitWidth(inputValue.getType()) != 1) {
      repOp.emitError("ReplicateOp with input width != 1 is not supported");
      return failure();
    }

    OpBuilder builder(repOp);
    IRRewriter rewriter(builder);

    auto resultValue = repOp.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());
    

    if (resultValueWidth > 4) {
      SmallVector<Value> intermediateResults;
      for (auto&& [sigId, sigWidth]: split_signal_4B(resultValueWidth)) {
        auto resultType = rewriter.getIntegerType(sigWidth);
        auto newOp = rewriter.create<toucan::LUTOp>(repOp.getLoc(), resultType, toucan::LUTOpName::LUT_Rep1b, ValueRange({inputValue}));
        intermediateResults.push_back(newOp.getResult());
      }
      concat_4b_and_replace(repOp.getOperation(), repOp.getResult(), intermediateResults, rewriter);
    } else {
      auto resultType = rewriter.getIntegerType(resultValueWidth);
      auto newOp = rewriter.create<toucan::LUTOp>(repOp.getLoc(), resultType, toucan::LUTOpName::LUT_Rep1b, ValueRange({inputValue}));
      tryCopySVNameHint(repOp.getOperation(), newOp.getOperation());
      rewriter.replaceAllUsesWith(repOp, newOp);
    }

    return success();
  }


  // shift left, inputs are 4b values from msb to lsb
  Value dshlCore(SmallVector<Value> &inputs, Value shamt, IRRewriter &rewriter, Operation* op) {
//    if (inputs.size() > 1) {
//      auto lastValue = *inputs.end();
//      assert(hw::getBitWidth(lastValue.getType()) == 4);
//    }

    auto shamtWidth = hw::getBitWidth(shamt.getType());
    uint64_t possibleShifts = 1L << shamtWidth;

    auto int4BType = rewriter.getI4Type();
    auto zeroConst = rewriter.create<hw::ConstantOp>(op->getLoc(), int4BType, 0);
    auto zeroConstValue = zeroConst.getResult();

    uint64_t resultWidth = 0;
    for (auto each_section: inputs) {
        auto sectionBitWidth = hw::getBitWidth(each_section.getType());
        resultWidth += sectionBitWidth;
        assert(sectionBitWidth <= 4);
    }

    auto result_sections = (resultWidth + 3) / 4;
    auto input_sections = inputs.size();
    assert(result_sections == input_sections);

    // intermediateResult[m][n] is inputs[m] << n
    SmallVector<SmallVector<Value, 4>, 128> intermediateResults;
    for (size_t i = 0; i < inputs.size(); i++) {
      // Shift by 1, 2, and 3
      auto op1 = inputs[i];
      auto op2 = (i < inputs.size() - 1) ? inputs[i+1] : zeroConstValue;

      auto sh1Op = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Shl1, op1, op2);
      auto sh2Op = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Shl2, op1, op2);
      auto sh3Op = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Shl3, op1, op2);

      intermediateResults.push_back({op1, sh1Op.getResult(), sh2Op.getResult(), sh3Op.getResult()});
    }


    SmallVector<Value> shiftResult;
    for (size_t shamt_static = 0; shamt_static < possibleShifts; shamt_static++) {
      // Result of shift by i
      auto shamt_real = shamt_static % 4;
      auto shamt_filling = shamt_static - shamt_real;
      auto filling_sections = shamt_filling / 4;

      if ((result_sections < filling_sections)) {
        // if shamt_static is too large
        filling_sections = result_sections;
      }

      SmallVector<Value> tempResult;
      for (int j = (result_sections - filling_sections - 1); j >= 0; j--) {
        auto this_section_value = intermediateResults[j][shamt_real];
        tempResult.push_back(this_section_value);
      }
      // Fill trailing zeros
      for (int j = filling_sections - 1; j >= 0; j--) {
        tempResult.push_back(zeroConstValue);
      }

      auto concatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), tempResult);
      auto concatValue = concatOp.getResult();
      auto concatValueWidth = static_cast<uint64_t>(hw::getBitWidth(concatValue.getType()));
      assert(concatValueWidth >= resultWidth);

      auto result = concatValue;
      if (concatValueWidth > resultWidth) {
        // remove some extra bits
        auto extractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), concatValue, 0, resultWidth);
        result = extractOp.getResult();
      }

      shiftResult.push_back(result);
    }

    // Finally, select result using muxes
    auto result = generate_mux_chain(op, rewriter, shiftResult, shamt);

    assert(static_cast<uint64_t>(hw::getBitWidth(result.getType())) == resultWidth);
    
    return result;

  }

  // shift right, inputs need to be extended!!!, inputs are 4b values from msb to lsb
  Value dshrCore(IRRewriter &rewriter, Operation* op, SmallVector<Value> &inputs, uint64_t resultWidth, Value shamt, Value fillingValue) {

    auto shamtWidth = hw::getBitWidth(shamt.getType());
    uint64_t possibleShifts = 1L << shamtWidth;

    auto result_sections = (resultWidth + 3) / 4;
    auto input_sections = inputs.size();
    assert(result_sections == input_sections);
    assert(result_sections > 0);

    for (auto each_section: inputs) {
        auto sectionBitWidth = hw::getBitWidth(each_section.getType());
        // inputTotalBitWidth += sectionBitWidth;
        // Unlike dshl, here we require input aligned to 4b, to avoid complexity of signed extention here
        assert(sectionBitWidth == 4);
    }

    // intermediateResult[m][n] is inputs[m] >> n
    SmallVector<SmallVector<Value, 4>, 128> intermediateResults;
    for (size_t i = 0; i < inputs.size(); i++) {
      // Shift by 1, 2, and 3
      auto op1 = (i == 0) ? fillingValue : inputs[i-1];
      auto op2 = inputs[i];

      auto sh1Op = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Shr1, op1, op2);
      auto sh2Op = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Shr2, op1, op2);
      auto sh3Op = rewriter.create<toucan::LUTOp>(op->getLoc(), toucan::LUTOpName::LUT_Shr3, op1, op2);

      intermediateResults.push_back({op2, sh1Op.getResult(), sh2Op.getResult(), sh3Op.getResult()});
      assert(intermediateResults.size() < 10000);
    }
    assert(intermediateResults.size() == result_sections);


    SmallVector<Value> shiftResult;
    for (size_t shamt_static = 0; shamt_static < possibleShifts; shamt_static++) {
      // Result of shift by i
      auto shamt_real = shamt_static % 4;
      auto shamt_filling = shamt_static - shamt_real;
      auto filling_sections = shamt_filling / 4;

      if ((result_sections < filling_sections)) {
        // if shamt_static is too large
        filling_sections = result_sections;
      }

      SmallVector<Value> tempResult;
      // Fill heading 4bs
      for (size_t j = 0; j < filling_sections; j++) {
        tempResult.push_back(fillingValue);
        assert(tempResult.size() < 10000);
      }
      for (int j = result_sections - 1; j >= static_cast<int>(filling_sections); j--) {
        auto this_section_value = intermediateResults[j][shamt_real];
        tempResult.push_back(this_section_value);
        assert(tempResult.size() < 10000);
      }

      auto concatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), tempResult);
      auto concatValue = concatOp.getResult();
      auto concatValueWidth = static_cast<uint64_t>(hw::getBitWidth(concatValue.getType()));
      assert(concatValueWidth >= resultWidth);

      auto result = concatValue;
      if (concatValueWidth > resultWidth) {
        // remove some extra bits
        assert(concatValueWidth - resultWidth < 4);
        auto extractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), concatValue, 0, resultWidth);
        result = extractOp.getResult();
      }

      shiftResult.push_back(result);
    }

    // Finally, select result using muxes
    auto result = generate_mux_chain(op, rewriter, shiftResult, shamt);

    assert(static_cast<uint64_t>(hw::getBitWidth(result.getType())) == resultWidth);
    
    return result;

  }

  // This function may modify shamtValue!
  // For now, don't check if shamt has extra bits. leave it to canonicalizer.
  size_t limitShamtWidth(Value &inputValue, Value &shamtValue, IRRewriter &rewriter, Operation *op) {
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

  LogicalResult splitCombShl(comb::ShlOp &shlOp) {

    OpBuilder builder(shlOp);
    IRRewriter rewriter(builder);

    auto inputValue = shlOp.getLhs();
    auto shamtValue = extractMinimumWidth(shlOp.getRhs(), rewriter, shlOp.getOperation());
    auto shamtWidth = hw::getBitWidth(shamtValue.getType());
    if (shamtWidth > 30) {
      shlOp.emitError("shl: Shift amount is too large: " + std::to_string(shamtWidth));
      return failure();
    }

    // shamtWidth = limitShamtWidth(inputValue, shamtValue, rewriter, shlOp.getOperation());
    assert(hw::getBitWidth(shamtValue.getType()) == shamtWidth);

    // Split input signals
    auto inputValues_4b = split_value_4B(shlOp.getOperation(), inputValue, rewriter);

    auto result = dshlCore(inputValues_4b, shamtValue, rewriter, shlOp.getOperation());

    auto oldResult = shlOp.getResult();

    auto oldResultWidth = hw::getBitWidth(oldResult.getType());
    auto newResultWidth = hw::getBitWidth(result.getType());
    assert(oldResultWidth == newResultWidth);

    rewriter.replaceAllUsesWith(oldResult, result);

    return success();
  }

  LogicalResult splitCombShrU(comb::ShrUOp &shruOp) {

    OpBuilder builder(shruOp);
    IRRewriter rewriter(builder);

    auto inputValue = shruOp.getLhs();
    auto shamtValue = extractMinimumWidth(shruOp.getRhs(), rewriter, shruOp.getOperation());
    auto shamtWidth = hw::getBitWidth(shamtValue.getType());
    if (shamtWidth > 30) {
      shruOp.emitError("shru: Shift amount is too large: " + std::to_string(shamtWidth));
      return failure();
    }

    auto inputValueWidth = hw::getBitWidth(inputValue.getType());
    assert(inputValueWidth > 1);

    // shamtWidth = limitShamtWidth(inputValue, shamtValue, rewriter, shruOp.getOperation());
    assert(hw::getBitWidth(shamtValue.getType()) == shamtWidth);

    // Split input signals
    auto inputValues_4b = split_value_4B(shruOp.getOperation(), inputValue, rewriter);

    auto headFragmentWidth = hw::getBitWidth(inputValues_4b[0].getType());
    if (headFragmentWidth != 4) {
      assert(headFragmentWidth < 4);
      // Since it's ShrU, simply fill head with 0
      auto paddingBits = 4 - headFragmentWidth;
      auto dataType = rewriter.getIntegerType(paddingBits);
      auto zeroConstOp = rewriter.create<hw::ConstantOp>(shruOp.getLoc(), dataType, 0);
      auto zeroConstValue = zeroConstOp.getResult();

      auto concatOp = rewriter.create<comb::ConcatOp>(shruOp.getLoc(), zeroConstValue, inputValues_4b[0]);

      auto concatValue = concatOp.getResult();
      assert(hw::getBitWidth(concatValue.getType()) == 4);

      inputValues_4b[0] = concatValue;
    }

    auto zeroConstOp = rewriter.create<hw::ConstantOp>(shruOp.getLoc(), rewriter.getI4Type(), 0);
    auto zeroConstValue = zeroConstOp.getResult();

    auto result = dshrCore(rewriter, shruOp.getOperation(), inputValues_4b, inputValueWidth, shamtValue, zeroConstValue);

    auto oldResult = shruOp.getResult();

    auto oldResultWidth = hw::getBitWidth(oldResult.getType());
    auto newResultWidth = hw::getBitWidth(result.getType());
    assert(oldResultWidth == newResultWidth);

    rewriter.replaceAllUsesWith(oldResult, result);

    return success();
  }

  LogicalResult splitCombShrS(comb::ShrSOp &shrsOp) {

    OpBuilder builder(shrsOp);
    IRRewriter rewriter(builder);

    auto inputValue = shrsOp.getLhs();
    auto shamtValue = extractMinimumWidth(shrsOp.getRhs(), rewriter, shrsOp.getOperation());
    auto shamtWidth = hw::getBitWidth(shamtValue.getType());
    if (shamtWidth > 30) {
      shrsOp.emitError("shrs: Shift amount is too large: " + std::to_string(shamtWidth));
      return failure();
    }

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

      auto concatOp = rewriter.create<comb::ConcatOp>(shrsOp.getLoc(), fragmentFillingValue, inputValues_4b[0]);

      inputValues_4b[0] = concatOp.getResult();
    }

    auto result = dshrCore(rewriter, shrsOp.getOperation(), inputValues_4b, inputValueWidth, shamtValue, fillingValue);

    auto oldResult = shrsOp.getResult();

    auto oldResultWidth = hw::getBitWidth(oldResult.getType());
    auto newResultWidth = hw::getBitWidth(result.getType());
    assert(oldResultWidth == newResultWidth);

    rewriter.replaceAllUsesWith(oldResult, result);

    return success();
  }

  void emitUnsupportOpError(Operation *op) {
    op->emitError("Operation not supported");
  }

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    for (auto &stmt: mod.getOps()) {
      // Lower hw.constant
      if (auto constOp = dyn_cast<hw::ConstantOp>(stmt)) {
        // dont do anything with const. It may be folded
        // if (failed(splitHWConstOp(constOp))) return failure();
      } else if (auto andOp = dyn_cast<comb::AndOp>(stmt)) {
        // Lower comb.and
        if (failed(splitCombAndOp(andOp))) return failure();
        toRemove.push_back(andOp);
      } else if (auto orOp = dyn_cast<comb::OrOp>(stmt)) {
        // Lower comb.or
        if (failed(splitCombOrOp(orOp))) return failure();
        toRemove.push_back(orOp);
      } else if (auto xorOp = dyn_cast<comb::XorOp>(stmt)) {
        // Lower comb.xor
        if (failed(splitCombXorOp(xorOp))) return failure();
        toRemove.push_back(xorOp);
      } else if (auto repOp = dyn_cast<comb::ReplicateOp>(stmt)) {
        // lower replicate
        if (failed(splitCombReplicate(repOp))) return failure();
        toRemove.push_back(repOp);
      } else if (auto shlOp = dyn_cast<comb::ShlOp>(stmt)) {
        if (failed(splitCombShl(shlOp))) return failure();
        toRemove.push_back(shlOp);
      } else if (auto shruOp = dyn_cast<comb::ShrUOp>(stmt)) {
        if (failed(splitCombShrU(shruOp))) return failure();
        toRemove.push_back(shruOp);
      } else if (auto shrsOp = dyn_cast<comb::ShrSOp>(stmt)) {
        if (failed(splitCombShrS(shrsOp))) return failure();
        toRemove.push_back(shrsOp);
      } else if (auto icmpOp = dyn_cast<comb::ICmpOp>(stmt)) {
        // TODO: lower icmp
      } else if (auto mulOp = dyn_cast<comb::MulOp>(stmt)) {
        // TODO: lower mul
      } else if (
        isa<comb::MuxOp>(stmt) ||
        isa<comb::AddOp>(stmt) ||
        isa<comb::SubOp>(stmt)
      ) {
        // Will be handled by next pass
        // do nothing
      } else if (
        isa<comb::DivSOp>(stmt) ||
        isa<comb::DivUOp>(stmt) ||
        isa<comb::ModSOp>(stmt) ||
        isa<comb::ModUOp>(stmt)
      ) {
        emitUnsupportOpError(&stmt);
      }

    }

    if (!toRemove.empty()) {
      LLVM_DEBUG(
        char buffer[128];
        format("Removing %d Ops\n", toRemove.size()).snprint(buffer, 128);
        llvm::dbgs() << buffer
        );
      for (auto op: toRemove) op->erase();
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
  }

};

std::unique_ptr<mlir::Pass> toucan::createLowerCombTo4B_1Pass() {
  return std::make_unique<LowerCombTo4B_1Pass>();
}
