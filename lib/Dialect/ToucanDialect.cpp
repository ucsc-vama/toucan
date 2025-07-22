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
#include "llvm/ADT/APInt.h"
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
#include "toucan/ToucanVecOpLimits.h"

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
    case LUTOpName::LUT_Rep1b:
    case LUTOpName::LUT_XorR:
    case LUTOpName::LUT_Nop:
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
    case LUTOpName::LUT_Cmp_Ult:
    case LUTOpName::LUT_Cmp_Slt4b:
    case LUTOpName::LUT_Add:
    case LUTOpName::LUT_Sub:
      return 2;
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

  switch (getOpName()) {
    case LUTOpName::LUT_Rep1b: {
      if (getInputs().size() != 1) {
        return emitError("Rep1b should have only 1 input");
      }
      if (hw::getBitWidth(getInputs().front().getType()) != 1) {
        return emitError("Rep1b input should only be 1 bit");
      }
      break;
    }

    default: {}
  }

  return success();
}

LogicalResult StaticVectorSegmentReadOp::verify() {
  auto vecHandleDeclOp = getHandle().getDefiningOp();
  if (!isa<toucan::VectorArithOp>(vecHandleDeclOp)) {
    return emitError("StaticVectorSegmentRead should only accept vector from VectorArithOp");
  }
  auto vecElemWidth = getHandle().getType().getElementWidth();
  if (vecElemWidth != 4) {
    return emitError("StaticVectorSegmentRead only accept vector with every element width is 4");
  }
  auto resultWidth = hw::getBitWidth(getResult().getType());
  if (resultWidth != 4) {
    return emitError("StaticVectorSegmentRead should only return a value with width of 4");
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

LogicalResult MemWriteOp::verify() {
  auto addrVecDeclOp = getAddrVec().getDefiningOp();
  if (isa<toucan::DefConstVectorOp>(addrVecDeclOp)) {
    return emitError() << "For now does not support write to a constant memory address";
  }
  auto vecLength = getAddrVec().getType().getLength();

  if (vecLength != 8) {
    return emitError() << "MemWrite addrVec should have length of 8, but got " << vecLength;
  }
  return success();
}

LogicalResult MemReadOp::verify() {
  auto addrVecDeclOp = getAddrVec().getDefiningOp();
  if (isa<toucan::DefConstVectorOp>(addrVecDeclOp)) {
    return emitError() << "For now does not support read from a constant memory address";
  }
  auto vecLength = getAddrVec().getType().getLength();

  if (vecLength != 8) {
    return emitError() << "MemRead addrVec should have length of 8, but got " << vecLength;
  }

  auto vecElemWidth = getAddrVec().getType().getElementWidth();
  if (vecElemWidth != 4) {
    return emitError() << "MemRead addrVec should have element width of 4";
  }
  return success();
}

LogicalResult VectorArithOp::verify() {
  auto v1Length = getV1().getType().getLength();
  auto v1Width = getV1().getType().getElementWidth();

  auto v2Length = getV2().getType().getLength();
  auto v2Width = getV2().getType().getElementWidth();

  if (v1Width != 4) {
    return emitError() << "For now only support 4 bit vector ops!";
  }
  
  if (v1Length != v2Length) {
    return emitError() << "Vector arith should have identical input vectors";
  }

  if (v1Width != v2Width) {
    return emitError() << "Vector arith should have identical input vectors";
  }

  if (v1Width != 4) {
    return emitError() << "Vector arith only accepts vector with element width of 4";
  }

  if ((v1Width * v1Length) > TOUCAN_VEC_OP_MAX_WIDTH) {
    return emitError("Vector width is too large.");
  }

  return success();
}

LogicalResult VectorLogicOp::verify() {
  auto v1Length = getV1().getType().getLength();
  auto v1Width = getV1().getType().getElementWidth();

  auto v2Length = getV2().getType().getLength();
  auto v2Width = getV2().getType().getElementWidth();
  
  if (v1Length != v2Length) {
    return emitError() << "Vector arith should have identical input vectors";
  }

  if (v1Width != v2Width) {
    return emitError() << "Vector arith should have identical input vectors";
  }

  if (v1Width != 4) {
    return emitError() << "Vector arith only accepts vector with element width of 4";
  }

  if ((v1Width * v1Length) > TOUCAN_VEC_OP_MAX_WIDTH) {
    return emitError("Vector width is too large.");
  }

  return success();
}

size_t LUTOp::getResultWidth1(toucan::LUTOpName opName, ValueRange inputs) {
  switch (opName) {
    case LUTOpName::LUT_Rep1b: return 4;
    case LUTOpName::LUT_XorR: return 1;
    default:
      break;
  }
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

    case LUTOpName::LUT_Cmp_Eq: 
    case LUTOpName::LUT_Cmp_Ult:
    case LUTOpName::LUT_Cmp_Slt4b:
    return 1;

    case LUTOpName::LUT_Sub:
    case LUTOpName::LUT_Add: {
      // Note: Add and sub always returns 4 bits. Use extract to limit output width
      return 4;
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


static inline LogicalResult canonicalize_LUT_Add(LUTOp &op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();

  auto lhs = inputs[0];
  auto rhs = inputs[1];

  auto lhs_isConstZero = value_is_const_zero(lhs);
  auto rhs_isConstZero = value_is_const_zero(rhs);

  if (lhs_isConstZero && rhs_isConstZero) {
    rewriter.replaceOp(op, lhs);
    return success();
  } else if (lhs_isConstZero) {
    rewriter.replaceOp(op, rhs);
    return success();
  } else if (rhs_isConstZero) {
    rewriter.replaceOp(op, lhs);
    return success();
  }
  return failure();
}


LogicalResult LUTOp::canonicalize(LUTOp op, PatternRewriter &rewriter) {
  auto opName = op.getOpName();
  auto inputs = op.getInputs();


  switch(opName) {
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
      assert(hw::getBitWidth(inputs[0].getType()) == hw::getBitWidth(inputs[1].getType()) && "Input should of same type.");
      for (size_t i = 0; i < 2; i++) {
        auto opRand = inputs[i];
        auto opRand_another = inputs[1 - i];
        if (value_is_const_zero(opRand)) {
          rewriter.replaceOp(op, opRand_another);
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
      break;
    case LUTOpName::LUT_Add: {
      return canonicalize_LUT_Add(op, rewriter);
    }
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
    case LUTOpName::LUT_XorR:
    case LUTOpName::LUT_Nop:
    case LUTOpName::LUT_Cmp_Eq:
    case LUTOpName::LUT_Cmp_Ult:
    case LUTOpName::LUT_Cmp_Slt4b:
    case LUTOpName::LUT_Sub:
    break;
  }
  return failure();
}

// Note: Some op has exact same truth table with others.
toucan::LUTOpName LUTOp::getMappedOpName(toucan::LUTOpName opName) {
  switch (opName) {
    case LUTOpName::LUT_Shr1: return LUTOpName::LUT_Shl3;
    case LUTOpName::LUT_Shr2: return LUTOpName::LUT_Shl2;
    case LUTOpName::LUT_Shr3: return LUTOpName::LUT_Shl1;
    default: return opName;
  }
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


