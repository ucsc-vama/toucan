#include "toucan/ToucanDialect.h"
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

LogicalResult PrintOp::canonicalize(PrintOp op, PatternRewriter &rewriter) {
  auto enSignal = op.getEn();
  if (auto constOp = dyn_cast<hw::ConstantOp>(enSignal.getDefiningOp())) {
    auto constVal = constOp.getValue();
    if (!constVal.getBoolValue()) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

LogicalResult StopOp::canonicalize(StopOp op, PatternRewriter &rewriter) {
  auto enSignal = op.getEn();
  if (auto constOp = dyn_cast<hw::ConstantOp>(enSignal.getDefiningOp())) {
    auto constVal = constOp.getValue();
    if (!constVal.getBoolValue()) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}