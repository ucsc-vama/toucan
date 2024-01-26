#include "toucan/ToucanDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/ValueRange.h"
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
#include "llvm/Support/MathExtras.h"
#include <numeric>
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace mlir;
using namespace llvm;
using namespace toucan;

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

