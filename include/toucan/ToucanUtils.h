#pragma once

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/OM/OMDialect.h"

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

namespace toucan {
  mlir::StringRef getSVNameHitRef();

  std::optional<mlir::StringAttr> getSVNameHintAttr(mlir::Operation* op);

  void setSVNameHintAttr(mlir::Operation *op, mlir::StringAttr &value);

  const char* getRegNextSuffix();

  mlir::StringRef getSignalFragmentIDRef();

  void setSignalFragmentIDAttr(mlir::Operation *op, mlir::IntegerAttr &id);

  std::optional<mlir::IntegerAttr> getSignalFragmentIDAttr(mlir::Operation *op);




  std::vector<std::tuple<int, int>> split_signal_4B(int bit_width);
};