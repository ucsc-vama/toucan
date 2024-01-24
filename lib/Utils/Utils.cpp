
#include "mlir/IR/BuiltinAttributes.h"
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

  StringRef getSignalFragmentIDRef() {
    return "signal_fragment_id";
  }

  void setSignalFragmentIDAttr(Operation *op, IntegerAttr &id) {
    op->setAttr(getSignalFragmentIDRef(), id);
  }

  std::optional<IntegerAttr> getSignalFragmentIDAttr(Operation *op) {
    if (op->hasAttr(getSignalFragmentIDRef())) {
        return op->getAttrOfType<IntegerAttr>(getSignalFragmentIDRef());
    }
    return std::optional<IntegerAttr>();
  }

  std::vector<std::tuple<int, int>> split_signal_4B(int bit_width) {
    assert(bit_width > 0);
    std::vector<std::tuple<int, int>> chunks;
    int chunk_id = 0;
    int remaining_bits = bit_width;

    while (remaining_bits > 0) {
        int chunk_size = std::min(remaining_bits, 4);
        chunks.push_back({chunk_id++, chunk_size});
        remaining_bits -= chunk_size;
    }
    std::reverse(chunks.begin(), chunks.end());

    return chunks;
}
}

