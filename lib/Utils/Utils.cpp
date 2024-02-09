
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "toucan/ToucanUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
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

  void tryCopySVNameHint(mlir::Operation *from, mlir::Operation *to) {
    auto namehint = getSVNameHintAttr(from);
    if (namehint) {
      setSVNameHintAttr(to, namehint.value());
    }
  }

  void copyCustomizedAttrs(mlir::Operation *from, mlir::Operation *to) {
    auto namehint = getSVNameHintAttr(from);
    if (namehint) {
      setSVNameHintAttr(to, namehint.value());
    }
    auto fragmentId = getSignalFragmentIDAttr(from);
    if (fragmentId) {
      setSignalFragmentIDAttr(to, fragmentId.value());
    }
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

  llvm::SmallVector<mlir::Value> split_value_4B(mlir::Operation *op, const mlir::Value &value, mlir::RewriterBase &rewriter) {
    llvm::SmallVector<mlir::Value> ret;

    // TODO: this need improvement. Need consider sv namehint and rename this method
    auto valDefiningOp = value.getDefiningOp();
    if (valDefiningOp) {
      if (auto concatOp = llvm::dyn_cast<comb::ConcatOp>(valDefiningOp)) {
        auto inputs = concatOp.getInputs();
        assert(inputs.size() >= 2);
        if (hw::getBitWidth(inputs[0].getType()) <= 4) {
          for (size_t i = 1; i < inputs.size(); i++) ret.push_back(inputs[i]);
          if (llvm::all_of(ret, [&](Value &v){return hw::getBitWidth(v.getType()) == 4;})) {
            ret.insert(ret.begin(), inputs[0]);
            return ret;
          }
        }
      }
    }
    ret.clear();

    auto inputBitWidth = hw::getBitWidth(value.getType());

    if (inputBitWidth > 4) {
        auto chunks = split_signal_4B(inputBitWidth);
        for (auto [chunkId, chunkWidth]: chunks) {

            auto extractOp = rewriter.create<comb::ExtractOp>(op->getLoc(), value, chunkId * 4, chunkWidth);
            auto newValue_4b = extractOp.getResult();

            ret.push_back(newValue_4b);
        }
    } else {
      ret.push_back(value);
    }
    // std::reverse(ret.begin(), ret.end());
    return ret;
  }

  void concat_4b_and_replace(mlir::Operation *op, mlir::Value opResult, llvm::SmallVector<mlir::Value> &values, mlir::RewriterBase &rewriter) {
    if (values.size() > 1) {
      auto bitConcatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), values);
      tryCopySVNameHint(op, bitConcatOp.getOperation());
      rewriter.replaceOp(opResult.getDefiningOp(), bitConcatOp);
    } else {
       rewriter.replaceOp(opResult.getDefiningOp(), values[0]);
    }
      
  }

  mlir::Value generate_mux_chain(mlir::Operation *op, mlir::RewriterBase &rewriter, llvm::SmallVector<mlir::Value> values, mlir::Value index) {
    SmallVector<mlir::Value> outputs, inputs;
    inputs.append(values.begin(), values.end());
    auto elemType = values[0].getType();

    // Use 0 as default value
    auto defaultValueAttr = rewriter.getIntegerAttr(elemType, 0);
    auto defaultValueOp = rewriter.create<hw::ConstantOp>(op->getLoc(), defaultValueAttr);
    auto defaultValue = defaultValueOp.getResult();

    auto indexBits = hw::getBitWidth(index.getType());

    assert(indexBits < 20);
    assert((1L << indexBits) >= values.size());


    for (int64_t level = 0; level < indexBits; level++) {
      assert(inputs.size() > 1);
      outputs.clear();

      // Extract En signal for all muxes at this level
      auto addrFragmentOp = rewriter.create<comb::ExtractOp>(op->getLoc(), index, level, 1);
      auto addrFragment = addrFragmentOp.getResult();

      for (size_t mux_id = 0; mux_id < ((inputs.size() + 1) >> 1); mux_id++) {
        // fval: low addr, tval: high addr
        size_t val_id = mux_id << 1;
        auto fVal = (val_id <= inputs.size()) ? inputs[val_id] : defaultValue;
        val_id++;
        auto tVal = (val_id <= inputs.size()) ? inputs[val_id] : defaultValue;

        auto muxOp = rewriter.create<comb::MuxOp>(op->getLoc(), addrFragment, tVal, fVal);

        outputs.push_back(muxOp.getResult());
        assert(outputs.size() < 10000);
      }

      std::swap(inputs, outputs);
    }
    assert(inputs.size() == 1);

    auto resultValue = inputs.front();

    return resultValue;
  }


  bool value_is_const_zero(mlir::Value &inputVal) {
    if (auto constOp = inputVal.getDefiningOp<hw::ConstantOp>()) {
      if (constOp.getValue().isZero()) {
        return true;
      }
    }
    return false;
  }

  bool value_is_const_ones(mlir::Value &inputVal) {
    if (auto constOp = inputVal.getDefiningOp<hw::ConstantOp>()) {
      if (constOp.getValue().isAllOnes()) {
        return true;
      }
    }
    return false;
  }

  bool value_is_const(mlir::Value &inputVal) {
    if (auto constOp = inputVal.getDefiningOp<hw::ConstantOp>()) {
      return true;
    }
    return false;
  }

  mlir::Value extractMinimumWidth(mlir::Value val, mlir::RewriterBase &rewriter, mlir::Operation* op) {
    if (auto concatOp = val.getDefiningOp<comb::ConcatOp>()) {
      auto inputs = concatOp.getInputs();
      llvm::SmallVector<mlir::Value> minInputs;
      if (inputs.size() > 0) {
        bool doneMerging = false;
        for (size_t i = 0; i < inputs.size(); i++) {
          auto inputVal = inputs[i];

          if (!doneMerging) {
            if (auto constOp = inputVal.getDefiningOp<hw::ConstantOp>()) {
              auto constVal = constOp.getValue();
              auto leadingZeros = constVal.countLeadingZeros();
              auto constValWidth = constVal.getBitWidth();

              if (leadingZeros != constValWidth) {
                auto usefulBits = constValWidth - leadingZeros;
                auto truncatedVal = constVal.extractBits(usefulBits, 0);
                assert(constVal == truncatedVal.zext(constValWidth));
                
                auto newConstOp = rewriter.create<hw::ConstantOp>(op->getLoc(), truncatedVal);
                minInputs.push_back(newConstOp.getResult());
              }
            } else {
              if (i == 0) {
                // Cannot merge the first element. simply return
                return val;
              }
              doneMerging = true;
              minInputs.push_back(inputVal);
            }
          } else {
            minInputs.push_back(inputVal);
          }
          
        }

        // special case
        if (minInputs.size() == 1) {
          return minInputs[0];
        }

        auto newConcatOp = rewriter.create<comb::ConcatOp>(op->getLoc(), minInputs);
        return newConcatOp.getResult();
        
      }
    }
    return val;
  }


  mlir::Value padding_with_0_and_align_4b(mlir::Operation *op, mlir::RewriterBase &rewriter, mlir::Value val) {
    auto inputValueWidth = hw::getBitWidth(val.getType());

    auto paddingTargetWidth = ((inputValueWidth + 3) & (~0x3));

    if (paddingTargetWidth != inputValueWidth) {
      auto constOp = rewriter.create<hw::ConstantOp>(op->getLoc(), rewriter.getIntegerType(paddingTargetWidth - inputValueWidth), 0);
      auto constVal = constOp.getResult();

      auto paddingOp = rewriter.create<comb::ConcatOp>(op->getLoc(), ValueRange({constVal, val}));
      auto paddingValue = paddingOp.getResult();

      assert(hw::getBitWidth(paddingValue.getType()) % 4 == 0);

      return paddingValue;
    } else {
      return val;
    }
  }

    // mlir::Value generate_reduce_tree(mlir::RewriterBase rewritter, llvm::SmallVector<mlir::Value> inputs, mlir::Value fillingVal, std::function<mlir::Value(mlir::RewriterBase&, mlir::Value, mlir::Value)> cb) {
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

