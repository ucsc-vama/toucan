
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "toucan/ToucanUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
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
#include <utility>

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

    llvm::SmallVector<mlir::Value> split_value_4B(mlir::Operation *op, mlir::Value &value, mlir::IRRewriter &rewriter) {
        llvm::SmallVector<mlir::Value> ret;

        auto inputBitWidth = hw::getBitWidth(value.getType());

        if (inputBitWidth > 4) {
            auto chunks = split_signal_4B(inputBitWidth);
            for (auto [chunkId, chunkWidth]: chunks) {

                auto extractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), value, chunkId * 4, chunkWidth);
                auto newValue_4b = extractOp.getResult();

                ret.push_back(newValue_4b);
            }
        }
        return ret;
    }

    void concat_4b_and_replace(mlir::Operation *op, mlir::Value opResult, llvm::SmallVector<mlir::Value> &values, mlir::IRRewriter &rewriter) {
        auto bitConcatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), values);
        rewriter.replaceAllUsesWith(opResult, bitConcatOp);
    }

    // mlir::Value generate_reduce_tree(mlir::IRRewriter rewritter, llvm::SmallVector<mlir::Value> inputs, mlir::Value fillingVal, std::function<mlir::Value(mlir::IRRewriter&, mlir::Value, mlir::Value)> cb) {
    //     llvm::SmallVector<mlir::Value> outputs;
    //     auto levels = llvm::Log2_64_Ceil(inputs.size());
    //     for (size_t level = 0; level < levels; level++) {
    //         outputs.clear();
    //         for (size_t i = 0; i < ((inputs.size() + 1) >> 1); i++) {
    //             auto val_id = i << 1;
    //             auto op1 = (val_id <= inputs.size()) ? inputs[val_id] : fillingVal;
    //             val_id += 1;
    //             auto op2 = (val_id <= inputs.size()) ? inputs[val_id] : fillingVal;

    //             auto resultVal = cb(rewritter, op1, op2);
    //             outputs.push_back(resultVal);
    //         }
    //         std::swap(inputs, outputs);
    //     }
    //     return inputs.front();
    // }
}

