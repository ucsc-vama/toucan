
#include "toucan/ToucanUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

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

using namespace mlir;
using namespace circt;
using namespace toucan;

namespace toucan {
  StringRef getSVNameHitRef() {
    return "sv.namehint";
  }

  std::optional<StringAttr> getSVNameHintAttr(Operation* op) {
    if (op->hasAttr(getSVNameHitRef())) {
      return op->getAttrOfType<StringAttr>(getSVNameHitRef());
    }
    return std::optional<StringAttr>();
  }

  void setSVNameHintAttr(Operation *op, StringAttr &value) {
    op->setAttr(getSVNameHitRef(), value);
  }

  const char* getRegNextSuffix() {
    return "$next";
  }
}

