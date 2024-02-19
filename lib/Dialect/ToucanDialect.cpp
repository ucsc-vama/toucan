#include "toucan/ToucanDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/PatternMatch.h"

#include "toucan/ToucanDialect.cpp.inc"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace mlir;
using namespace llvm;
using namespace toucan;

#include "toucan/ToucanEnums.cpp.inc"

// #define GET_ATTRDEF_CLASSES
// #include "toucan/ToucanAttributes.cpp.inc"

void ToucanDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "toucan/Toucan.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "toucan/Toucan.cpp.inc"

#include "toucan/ToucanUtils.h"

LogicalResult PrintOp::canonicalize(PrintOp op, PatternRewriter &rewriter) {
  auto enSignal = op.getEn();
  if (auto constOp = enSignal.getDefiningOp<hw::ConstantOp>()) {
    auto constVal = constOp.getValue();
    if (!constVal.getBoolValue()) {
      // Erase current PrintOp if printOp.En == false (constant)
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}


LogicalResult MemReadOp::canonicalize(MemReadOp op, PatternRewriter &rewriter) {
  auto enSignal = op.getEn();
  if (auto constOp = enSignal.getDefiningOp<hw::ConstantOp>()) {
    auto constVal = constOp.getValue();
    if (!constVal.getBoolValue()) {
      // Erase current PrintOp if printOp.En == false (constant)
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

LogicalResult MemWriteOp::canonicalize(MemWriteOp op, PatternRewriter &rewriter) {
  auto enSignal = op.getEn();
  if (auto constOp = enSignal.getDefiningOp<hw::ConstantOp>()) {
    auto constVal = constOp.getValue();
    if (!constVal.getBoolValue()) {
      // Erase current PrintOp if printOp.En == false (constant)
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

LogicalResult StopOp::canonicalize(StopOp op, PatternRewriter &rewriter) {
  auto enSignal = op.getEn();
  if (auto constOp = enSignal.getDefiningOp<hw::ConstantOp>()) {
    auto constVal = constOp.getValue();
    if (!constVal.getBoolValue()) {
      // Erase current StopOp if stopOp.En == false (constant)
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

LogicalResult RegWriteOp::verify() {
  auto dataWidth = hw::getBitWidth(getData().getType());
  assert(dataWidth >= 0);
  auto regWidth = getReg().getType().getElementWidth();

  if (regWidth != static_cast<uint64_t>(dataWidth)) {
    return emitError() <<"Data width doesn't match register width! " << "Register width is " << regWidth << ", while data width is " << dataWidth;
  }
  return success();
}

size_t LUTOp::getLegalOperandCount(toucan::LUTOpName opName) {
  switch (opName) {
    case LUTOpName::LUT_Nop:
      return 0;
    case LUTOpName::LUT_Rep1b:
      return 1;
    case LUTOpName::LUT_And:
    case LUTOpName::LUT_Or:
    case LUTOpName::LUT_Xor:
    case LUTOpName::LUT_Shl1:
    case LUTOpName::LUT_Shl2:
    case LUTOpName::LUT_Shl3:
    case LUTOpName::LUT_Shr1:
    case LUTOpName::LUT_Shr2:
    case LUTOpName::LUT_Shr3:
    case LUTOpName::LUT_Cmp_Eq:
    case LUTOpName::LUT_Mul_Hi:
    case LUTOpName::LUT_Mul_Lo:
    case LUTOpName::LUT_Carry:
      return 2;
    case LUTOpName::LUT_Add:
    case LUTOpName::LUT_Mux:
    case LUTOpName::LUT_DShl:
    case LUTOpName::LUT_DShr:
      return 3;
  }
  llvm_unreachable("Unknow op name");
}

LogicalResult LUTOp::verify() {
  size_t legalOperandCount = getLegalOperandCount(getOpName()); 
  if (getInputs().size() != legalOperandCount) {
    return emitError() << "Unmatched oprand count for op " << stringifyLUTOpName(getOpName()) << ": expect " << legalOperandCount << ", got " << getInputs().size();
  }
  for (auto input: getInputs()) {
    if (hw::getBitWidth(input.getType()) > 4) {
      return emitError() << "Operand has a max width of 4";
    }
  }
  return success();
}

LogicalResult DefVectorOp::verify() {
  auto inputs = getInputs();

  if (inputs.empty()) {
    return emitError("Input vector is empty");
  }

  auto expectedElemWidth = hw::getBitWidth(inputs[0].getType());
  for (auto elem: inputs) {
    auto elemWidth = hw::getBitWidth(elem.getType());
    if (elemWidth != expectedElemWidth) {
      return emitError() << "Elements inside vector should have same width";
    }
  }

  return success();
}

size_t LUTOp::getResultWidth1(toucan::LUTOpName opName, ValueRange inputs) {
  assert(opName != toucan::LUTOpName::LUT_Rep1b && "LUT_Rep1b's output size should be given, instead of inffered");
  return hw::getBitWidth(inputs[0].getType());
}

size_t LUTOp::getResultWidth2(toucan::LUTOpName opName, ValueRange inputs) {
  switch (opName) {
    case LUTOpName::LUT_And:
    case LUTOpName::LUT_Or:
    case LUTOpName::LUT_Xor: 
    case LUTOpName::LUT_Shl1:
    case LUTOpName::LUT_Shl2:
    case LUTOpName::LUT_Shl3:
    case LUTOpName::LUT_Shr1:
    case LUTOpName::LUT_Shr2:
    case LUTOpName::LUT_Shr3: {
      assert(inputs.size() == 2);
      auto lhsSize = hw::getBitWidth(inputs[0].getType());
      auto rhsSize = hw::getBitWidth(inputs[1].getType());
      return std::max(lhsSize, rhsSize);
    }
    
    case LUTOpName::LUT_Mul_Hi:
    case LUTOpName::LUT_Mul_Lo: {
      return 4;
    } 
    
    case LUTOpName::LUT_Cmp_Eq:
    case LUTOpName::LUT_Carry: {
      return 1;
    }

    default: ;
  }
  llvm_unreachable(toucan::stringifyLUTOpName(opName).str().c_str());
}

size_t LUTOp::getResultWidth3(toucan::LUTOpName opName, ValueRange inputs) {
  if ((opName != toucan::LUTOpName::LUT_DShl) && (opName != toucan::LUTOpName::LUT_DShr)) {
    assert(hw::getBitWidth(inputs[0].getType()) == 1);
  } else {
    assert(hw::getBitWidth(inputs[0].getType()) <= 2);
  }

  auto lhsWidth = hw::getBitWidth(inputs[1].getType());
  auto rhsWidth = hw::getBitWidth(inputs[2].getType());

  return std::max(lhsWidth, rhsWidth);
}

LogicalResult LUTOp::canonicalize(LUTOp op, PatternRewriter &rewriter) {
  auto opName = op.getOpName();
  auto inputs = op.getInputs();
  // SmallVector<bool> opRandIsConst;
  // for (opRand: inputs) {
  //   opRandIsConst
  // }

  switch(opName) {
    case LUTOpName::LUT_Nop: {
      // Nop should not appear
      return failure();
    }
    case LUTOpName::LUT_And: {
      assert(inputs.size() == 2);
      for (size_t i = 0; i < 2; i++) {
        auto opRand = inputs[i];
        if (value_is_const_zero(opRand)) {
          auto constZeroOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getIntegerType(hw::getBitWidth(opRand.getType())), 0);
          rewriter.replaceOp(op, constZeroOp);
          return success();
        } else if (value_is_const_ones(opRand)) {
          rewriter.replaceOp(op, inputs[1-i]);
          return success();
        }
      }
      return failure();
    }
    case LUTOpName::LUT_Or: {
      assert(inputs.size() == 2);
      for (size_t i = 0; i < 2; i++) {
        auto opRand = inputs[i];
        if (value_is_const_zero(opRand)) {
          auto constZeroOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getIntegerType(hw::getBitWidth(opRand.getType())), 0);
          rewriter.replaceOp(op, constZeroOp);
          return success();
        } else if (value_is_const_ones(opRand)) {
          auto constOnesOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getIntegerType(hw::getBitWidth(opRand.getType())), -1);
          rewriter.replaceOp(op, constOnesOp);
          return success();
        }
      }
      return failure();
    }
    case LUTOpName::LUT_Xor: {
      assert(inputs.size() == 2);
      for (size_t i = 0; i < 2; i++) {
        auto opRand = inputs[i];
        if (value_is_const_zero(opRand)) {
          rewriter.replaceOp(op, inputs[1-i]);
          return success();
        }
      }
      return failure();
    }
    case LUTOpName::LUT_Rep1b:
    case LUTOpName::LUT_Shl1:
    case LUTOpName::LUT_Shl2:
    case LUTOpName::LUT_Shl3:
    case LUTOpName::LUT_Shr1:
    case LUTOpName::LUT_Shr2:
    case LUTOpName::LUT_Shr3:
    case LUTOpName::LUT_Cmp_Eq:
    case LUTOpName::LUT_Mul_Hi:
    case LUTOpName::LUT_Mul_Lo:
    case LUTOpName::LUT_Add: {
      // carry, lhs, rhs
      if (all_of(inputs, [&](auto val){return value_is_const_zero(val);})) {
        rewriter.replaceOp(op, inputs[1]);
        return success();
      }
      return failure();
    }
    case LUTOpName::LUT_Carry:
      break;
    case LUTOpName::LUT_Mux: {
      assert(inputs.size() == 3);
      auto enSignal = inputs[0];
      if (value_is_const_zero(enSignal)) {
        // Replace with fVal
        rewriter.replaceOp(op, inputs[2]);
        return success();
      }
      if (value_is_const_ones(enSignal)) {
        // Replace with tVal
        rewriter.replaceOp(op, inputs[1]);
        return success();
      }
      if (inputs[1] == inputs[2]) {
        rewriter.replaceOp(op, inputs[1]);
        return success();
      }
      return failure();
    }
    case LUTOpName::LUT_DShl:
    case LUTOpName::LUT_DShr:
    break;
  }
  return failure();
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APInt &value) {

  auto type = IntegerType::get(builder.getContext(), value.getBitWidth());
  auto attr = builder.getIntegerAttr(type, value);
  return build(builder, result, type, attr);
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       IntegerAttr value) {
  return build(builder, result, value.getType(), value);
}

/// This builder allows construction of small signed integers like 0, 1, -1
/// matching a specified MLIR IntegerType.  This shouldn't be used for general
/// constant folding because it only works with values that can be expressed in
/// an int64_t.  Use APInt's instead.
void ConstantOp::build(OpBuilder &builder, OperationState &result, Type type,
                       int64_t value) {
  auto numBits = type.cast<IntegerType>().getWidth();
  build(builder, result, APInt(numBits, (uint64_t)value, /*isSigned=*/true));
}


