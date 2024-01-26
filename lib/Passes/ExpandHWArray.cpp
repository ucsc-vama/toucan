
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>


#define GEN_PASS_DEF_EXPANDHWARRAY
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "ExpandHWArrayPass"

struct ExpandHWArrayPass : toucan::impl::ExpandHWArrayBase<ExpandHWArrayPass> {
  using ExpandHWArrayBase<ExpandHWArrayPass>::ExpandHWArrayBase;

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;


    for (auto &stmt: mod.getOps()) {
      if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(stmt)) {
        // An array
        auto arrayValue = arrayGetOp.getInput();
        auto arrayType = arrayValue.getType().cast<hw::ArrayType>();
        auto arrayNumElem = arrayType.getNumElements();
        auto arrayElemType = arrayType.getElementType();
        auto arrayIndex = arrayGetOp.getIndex();
        auto arrayIndexBits = hw::getBitWidth(arrayIndex.getType());


        OpBuilder builder(arrayGetOp);
        IRRewriter rewriter(builder);

        assert((1L << arrayIndexBits) >= arrayNumElem);
        assert(arrayNumElem > 1);
        assert(arrayIndexBits > 0);

        SmallVector<mlir::Value> values;
        if (auto constArrayOp = dyn_cast<hw::AggregateConstantOp>(arrayValue.getDefiningOp())) {
          // This array is a const array
          rewriter.setInsertionPointAfter(constArrayOp);

          auto constArrayValues = constArrayOp.getFields().getValue();
          for (auto &constArrayElem: constArrayValues) {
            if (!isa<mlir::IntegerAttr>(constArrayElem)) {
              constArrayOp->emitError() << "Only supports integer arrays!";
              return failure();
            }
            auto constArrayElemVal = cast<mlir::IntegerAttr>(constArrayElem);
            auto newConstOp = rewriter.create<hw::ConstantOp>(constArrayOp->getLoc(), constArrayElemVal);
            auto newConstValue = newConstOp.getResult();
            values.push_back(newConstValue);

            // toRemove.push_back(constArrayOp);
          }
        } else if (auto arrayCreateOp = dyn_cast<hw::ArrayCreateOp>(arrayValue.getDefiningOp())) {
          auto createdArrayValues = arrayCreateOp.getInputs();
          for (auto eachArrayValue: createdArrayValues) {
            values.push_back(eachArrayValue);
          }
          // toRemove.push_back(arrayCreateOp);
        } else {
          arrayGetOp->emitError() << "Accessing an array that is not defined by either hw::AggregateConstantOp or hw::ArrayCreateOp";
          return failure();
        }

        // Now we have values
        SmallVector<mlir::Value> current_level_inputs;
        SmallVector<mlir::Value> current_level_outputs;
        current_level_inputs.append(values.begin(), values.end());

        // Use 0 as default value
        auto defaultValueAttr = rewriter.getIntegerAttr(arrayElemType, 0);
        auto defaultValueOp = rewriter.create<hw::ConstantOp>(arrayGetOp->getLoc(), defaultValueAttr);
        auto defaultValue = defaultValueOp.getResult();

        rewriter.setInsertionPointAfterValue(arrayGetOp);

        for (int64_t level = 0; level < arrayIndexBits; level++) {
          assert(current_level_inputs.size() > 1);
          current_level_outputs.clear();

          // auto addrPos = arrayIndexBits - level;
          // Extract En signal for all muxes at this level
          auto addrFragmentOp = rewriter.create<comb::ExtractOp>(arrayGetOp.getLoc(), arrayIndex, level, 1);
          auto addrFragment = addrFragmentOp.getResult();

          for (size_t mux_id = 0; mux_id < ((current_level_inputs.size() + 1) >> 1); mux_id++) {
            // fval: low addr, tval: high addr
            size_t val_id = mux_id << 1;
            auto fVal = (val_id <= current_level_inputs.size()) ? current_level_inputs[val_id] : defaultValue;
            val_id++;
            auto tVal = (val_id <= current_level_inputs.size()) ? current_level_inputs[val_id] : defaultValue;

            auto muxOp = rewriter.create<comb::MuxOp>(arrayGetOp->getLoc(), addrFragment, tVal, fVal);

            current_level_outputs.push_back(muxOp.getResult());
          }

          std::swap(current_level_inputs, current_level_outputs);
        }
        assert(current_level_inputs.size() == 1);

        auto resultValue = current_level_inputs.front();
        auto resultDefiningOp = resultValue.getDefiningOp();

        auto namehint = getSVNameHintAttr(arrayGetOp);
        if (namehint) {
          setSVNameHintAttr(resultDefiningOp, namehint.value());
        }

        rewriter.replaceAllUsesWith(arrayGetOp, resultValue);

        toRemove.push_back(arrayGetOp);
      }

    }

    // For all hw.vector defining op, leave them to canonicalizer.
    for (auto op: llvm::reverse(toRemove)) op->erase();

    return success();
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<hw::HWModuleOp> modulesToProcess;
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }
    // // Sequential
    // for (auto mod: modulesToProcess) {
    //   auto ret = runOnModule(mod);
    //   if (failed(ret)) return signalPassFailure();
    // }

    // Parallel
    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result)) return signalPassFailure();
  }

};

std::unique_ptr<mlir::Pass> toucan::createExpandHWArrayPass() {
  return std::make_unique<ExpandHWArrayPass>();
}
