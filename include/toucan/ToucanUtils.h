#pragma once

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/OM/OMDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
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

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>


namespace toucan {

  bool value_is_const_zero(mlir::Value &inputVal);
  bool value_is_const_ones(mlir::Value &inputVal);
  bool value_is_const(mlir::Value &inputVal);

  mlir::StringRef getSVNameHitRef();

  std::optional<mlir::StringAttr> getSVNameHintAttr(mlir::Operation* op);

  void setSVNameHintAttr(mlir::Operation *op, mlir::StringAttr &value);

  void tryCopySVNameHint(mlir::Operation *from, mlir::Operation *to);

  const char* getRegNextSuffix();

  mlir::StringRef getSignalFragmentIDRef();
  
  bool hasSignalFragmentId(mlir::Operation *op);

  void setSignalFragmentIDAttr(mlir::Operation *op, mlir::IntegerAttr &id);

  std::optional<mlir::IntegerAttr> getSignalFragmentIDAttr(mlir::Operation *op);

  void copyCustomizedAttrs(mlir::Operation *from, mlir::Operation *to);


  mlir::StringRef getIOSignalMarkerAttrName();
  bool hasIOSignalMarker(mlir::Operation *op);
  void setIOSignalMarker(mlir::Operation *op);
  void removeIOSignalMarker(mlir::Operation *op);


  // mlir::StringRef getMemMaskFragmentIDAttr();
  // void setMemMaskFragmentIDAttr(mlir::Operation *op, mlir::IntegerAttr &id);
  // std::optional<mlir::IntegerAttr> getMemMaskFragmentIDAttr(mlir::Operation *op);

  mlir::StringRef getAccumulatedMemWidthAttr();
  void setAccumulatedMemWidthAttr(mlir::Operation *op, mlir::IntegerAttr &id);
  std::optional<mlir::IntegerAttr> getAccumulatedMemWidthAttr(mlir::Operation *op);




  std::vector<std::tuple<int, int>> split_signal_4B(int bit_width);

  llvm::SmallVector<mlir::Value> split_value_4B(mlir::Operation *op, const mlir::Value &value, mlir::RewriterBase &rewriter);

  void concat_4b_and_replace(mlir::Operation *op, mlir::Value opResult, llvm::SmallVector<mlir::Value> &values, mlir::RewriterBase &rewriter);

  void attachNameHintAndFragmentId(mlir::RewriterBase &rewriter, mlir::SmallVector<mlir::Value> &values, std::optional<mlir::StringAttr> namehint);

  void attachNameHintAndFragmentId(mlir::RewriterBase &rewriter, mlir::Value &value, std::optional<mlir::StringAttr> namehint);

  void attachNameHintAndFragmentId(mlir::RewriterBase &rewriter, mlir::Operation *op, std::optional<mlir::StringAttr> namehint);

  mlir::Value generate_mux_chain_for_array(mlir::Operation *op, mlir::RewriterBase &rewriter, llvm::SmallVector<mlir::Value> values, mlir::Value index);

  mlir::Value extractMinimumWidth(mlir::Value val, mlir::RewriterBase &rewriter, mlir::Operation* op);

  mlir::Value padding_with_0_and_align_4b(mlir::Operation *op, mlir::RewriterBase &rewriter, mlir::Value val);

  mlir::Value generate_reduce_tree(mlir::RewriterBase &rewritter, mlir::Location loc, llvm::SmallVector<mlir::Value> &inputs, std::function<mlir::Value(mlir::RewriterBase&, mlir::Location, mlir::Value, mlir::Value)> cb);

  mlir::Value removeHighBits(mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Value val, size_t bitsNeeded);

  bool isElementsFullWidth(mlir::OperandRange &vals);

  bool isElementsFullWidth(mlir::SmallVector<mlir::Value> &vals);
};