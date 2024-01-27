#pragma once

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/OM/OMDialect.h"

#include "mlir/IR/Value.h"
#include "toucan/ToucanDialect.h"


#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"


#include <tuple>
#include <iterator>
#include <functional>

namespace toucan {
  mlir::StringRef getSVNameHitRef();

  std::optional<mlir::StringAttr> getSVNameHintAttr(mlir::Operation* op);

  void setSVNameHintAttr(mlir::Operation *op, mlir::StringAttr &value);

  const char* getRegNextSuffix();

  mlir::StringRef getSignalFragmentIDRef();

  void setSignalFragmentIDAttr(mlir::Operation *op, mlir::IntegerAttr &id);

  std::optional<mlir::IntegerAttr> getSignalFragmentIDAttr(mlir::Operation *op);




  std::vector<std::tuple<int, int>> split_signal_4B(int bit_width);

  llvm::SmallVector<mlir::Value> split_value_4B(mlir::Operation *op, mlir::Value &value, mlir::IRRewriter &rewriter);

  void concat_4b_and_replace(mlir::Operation *op, mlir::Value opResult, llvm::SmallVector<mlir::Value> &values, mlir::IRRewriter &rewriter);

  // // template<class OpTy>
  // mlir::Value generate_reduce_tree(mlir::IRRewriter rewritter, llvm::SmallVector<mlir::Value> inputs, mlir::Value fillingVal, std::function<mlir::Value(mlir::IRRewriter&, mlir::Value, mlir::Value)> cb);
};