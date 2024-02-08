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

// LogicalResult BitScatterOp::verify() {
//   int64_t expectedInputWidth = 0;
//   for (auto elem: getOutputs()) {
//     expectedInputWidth += hw::getBitWidth(elem.getType());
//   }
//   if (expectedInputWidth != hw::getBitWidth(getInput().getType())) {
//     return emitError() << "Input and output width not match!";
//   }
//   return success();
// }

// void BitScatterOp::build(OpBuilder &odsBuilder, OperationState &odsState, mlir::Value input) {
//   // build
//   auto inputBitWidth = hw::getBitWidth(input.getType());
//   std::vector<mlir::Type> outputTypes;

//   auto chunks = split_signal_4B(inputBitWidth);
//   for (auto &chunk: llvm::reverse(chunks)) {
//     auto bitWidth = std::get<1>(chunk);
//     outputTypes.push_back(std::move(odsBuilder.getIntegerType(bitWidth)));
//   }

//   build(odsBuilder, odsState, mlir::ArrayRef<mlir::Type>(outputTypes), input);
// }


// LogicalResult BitAggregateOp::verify() {
//   int64_t expectedResultWidth = 0;
//   for (auto elem: getInputs()) {
//     expectedResultWidth += hw::getBitWidth(elem.getType());
//   }
//   if (expectedResultWidth != hw::getBitWidth(getOutput().getType())) {
//     return emitError() << "Input and output width not match!";
//   }
//   return success();
// }


