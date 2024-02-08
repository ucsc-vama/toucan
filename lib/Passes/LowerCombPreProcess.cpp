
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Attributes.h"
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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <memory>


#define GEN_PASS_DEF_LOWERCOMBPREPROCESS
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombPreProcessPass"

struct LowerCombPreProcessPass : toucan::impl::LowerCombPreProcessBase<LowerCombPreProcessPass> {
  using LowerCombPreProcessBase<LowerCombPreProcessPass>::LowerCombPreProcessBase;

  template<class OpTy>
  LogicalResult lowerOp(OpTy &op) {
    auto inputs = op.getInputs();
    if (inputs.size() > 2) {
      OpBuilder builder(op);
      IRRewriter rewriter(builder);

      auto head = inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
        auto currentVal = inputs[i];
        auto newInputs = ValueRange({head, currentVal});
        auto newOp = rewriter.create<OpTy>(op.getLoc(), newInputs, false);
        head = newOp.getResult();
      }

      rewriter.replaceAllUsesWith(op.getResult(), head);
      return success();
    }
    return failure();
  }

  LogicalResult lowerReplicateOp(comb::ReplicateOp &op) {
    // Handles repop with input width > 1
    auto inputValue = op.getInput();
    auto inputValueWidth = hw::getBitWidth(inputValue.getType());
    if (inputValueWidth == 1) {
      // Do nothing if width is 1
      return failure();
    }
    
    OpBuilder builder(op);
    IRRewriter rewriter(builder);

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    assert(resultValueWidth % inputValueWidth == 0);
    assert(resultValueWidth > inputValueWidth);

    auto numOf4B = static_cast<size_t>(resultValueWidth / inputValueWidth);


    SmallVector<Value> values;
    for (size_t i = 0; i < numOf4B; i++) {
      values.push_back(inputValue);
    }

    auto newOp = rewriter.create<comb::ConcatOp>(op.getLoc(), values);
    rewriter.replaceAllUsesWith(op, newOp);

    return success();
  }

  LogicalResult lowerShru_1b(comb::ShrUOp &op) {
    // Remove some 1 bit shru
    auto resultValue = op.getResult();
    auto resultWidth = hw::getBitWidth(resultValue.getType());

    if (resultWidth == 1) {
      OpBuilder builder(op);
      IRRewriter rewriter(builder);

      auto en = op.getRhs();
      auto data = op.getLhs();
      assert(hw::getBitWidth(en.getType()) == 1);
      assert(hw::getBitWidth(data.getType()) == 1);

      auto constZeroOp = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getI1Type(), 0);
      auto constZeroValue = constZeroOp.getResult();

      auto muxOp = rewriter.create<comb::MuxOp>(op.getLoc(), en, constZeroValue, data);

      rewriter.replaceAllUsesWith(op, muxOp);

      return success();
    }
    return failure();
  }

  LogicalResult lowerAggregateConsts(hw::AggregateConstantOp &constArrayOp) {
    OpBuilder builder(constArrayOp);
    IRRewriter rewriter(builder);

    SmallVector<SmallVector<Attribute>> array_values;

    auto constArrayValues = constArrayOp.getFields().getValue();

    auto elemWidth = hw::getBitWidth((constArrayValues.begin()->cast<mlir::IntegerAttr>().getType()));
    auto numChunks = (elemWidth + 3) / 4;
    for (size_t i = 0; i < static_cast<size_t>(numChunks); i++) {
      array_values.push_back({});
    }

    for (auto &constArrayElem: constArrayValues) {
      if (!isa<mlir::IntegerAttr>(constArrayElem)) {
        constArrayOp->emitError() << "Only supports integer arrays!";
        return failure();
      }

      auto constArrayElemVal = cast<mlir::IntegerAttr>(constArrayElem).getValue();
      auto constArrayElemValWidth = constArrayElemVal.getBitWidth();

      int startPos = (numChunks-1) * 4;
      for (auto [chunkId, chunkWidth]: (split_signal_4B(constArrayElemValWidth))) {
        if (startPos + chunkWidth > constArrayElemValWidth) {
          llvm::dbgs() << "Hit\n";
        }
        auto val = constArrayElemVal.extractBits(chunkWidth, startPos);
        startPos -= 4;
        auto intAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(val.getBitWidth()), val.getLimitedValue());
        array_values[chunkId].push_back(intAttr);
      }
    }

    SmallVector<Value> defVecHandles;
    for (auto elems: array_values) {
      auto arrayAttr = rewriter.getArrayAttr(elems);
      auto newDefVecOp = rewriter.create<toucan::DefConstVectorOp>(constArrayOp.getLoc(), arrayAttr);
      defVecHandles.push_back(newDefVecOp.getHandle());
    }

    for (auto userOp: constArrayOp->getUsers()) {
      if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(userOp)) {
        rewriter.setInsertionPointAfter(arrayGetOp);

        auto arrayIndex = extractMinimumWidth(arrayGetOp.getIndex(), rewriter, userOp);
        auto arrayIndexBits = hw::getBitWidth(arrayIndex.getType());

        auto maxIndexBits = llvm::Log2_64_Ceil(constArrayValues.size());
        if (arrayIndexBits > maxIndexBits) {
          arrayGetOp.emitWarning() << "Too much index bits than necessary. (Expect index bits no more than " << maxIndexBits << ", but got " << arrayIndexBits;
        }

        auto indicies_4b = split_value_4B(userOp, arrayIndex, rewriter);

        SmallVector<Value> arrayGetResults;
        for (auto vecHandle: defVecHandles) {
          auto readOp = rewriter.create<toucan::VectorReadOp>(arrayGetOp.getLoc(), vecHandle, indicies_4b);
          arrayGetResults.push_back(readOp);
        }

        auto concatOp = rewriter.create<comb::ConcatOp>(constArrayOp.getLoc(), arrayGetResults);

        copyCustomizedAttrs(userOp, concatOp);
        rewriter.replaceAllUsesWith(arrayGetOp, concatOp);

      } else {
        return userOp->emitError() << "Unknow op using const array";
      }
    }

    return success();

  }

  LogicalResult lowerArray(hw::ArrayCreateOp &arrayCreateOp) {
    OpBuilder builder(arrayCreateOp);
    IRRewriter rewriter(builder);

    SmallVector<SmallVector<Value>> array_values;

    auto arrayInputs = arrayCreateOp.getInputs();
    auto arrayLength = arrayInputs.size();
    auto elemWidth = hw::getBitWidth((arrayInputs[0].getType()));
    auto numChunks = (elemWidth + 3) / 4;
    for (size_t i = 0; i < static_cast<size_t>(numChunks); i++) {
      array_values.push_back({});
    }

    for (const auto &arrayElem: arrayInputs) {
      // auto arrayElemWidth = hw::getBitWidth(arrayElem.getType());
      auto chunkValues = split_value_4B(arrayCreateOp.getOperation(), arrayElem, rewriter);
      for (size_t i = 0; i < chunkValues.size(); i++) {
        array_values[i].push_back(chunkValues[i]);
      }
    }

    SmallVector<Value> defVecHandles;
    for (auto elems: array_values) {
      auto newDefVecOp = rewriter.create<toucan::DefVectorOp>(arrayCreateOp.getLoc(), elems);
      defVecHandles.push_back(newDefVecOp.getHandle());
    }

    for (auto userOp: arrayCreateOp->getUsers()) {
      if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(userOp)) {
        rewriter.setInsertionPointAfter(arrayGetOp);

        auto arrayIndex = extractMinimumWidth(arrayGetOp.getIndex(), rewriter, userOp);
        auto arrayIndexBits = hw::getBitWidth(arrayIndex.getType());

        auto maxIndexBits = llvm::Log2_64_Ceil(arrayLength);
        if (arrayIndexBits > maxIndexBits) {
          arrayGetOp.emitWarning() << "Too much index bits than necessary. (Expect index bits no more than " << maxIndexBits << ", but got " << arrayIndexBits;
        }

        auto indicies_4b = split_value_4B(userOp, arrayIndex, rewriter);

        SmallVector<Value> arrayGetResults;
        for (auto vecHandle: defVecHandles) {
          auto readOp = rewriter.create<toucan::VectorReadOp>(arrayGetOp.getLoc(), vecHandle, indicies_4b);
          arrayGetResults.push_back(readOp);
        }

        auto concatOp = rewriter.create<comb::ConcatOp>(arrayCreateOp.getLoc(), arrayGetResults);

        copyCustomizedAttrs(userOp, concatOp);
        rewriter.replaceAllUsesWith(arrayGetOp, concatOp);

      } else {
        return userOp->emitError() << "Unknow op using const array";
      }
    }

    return success();

  }

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    for (auto &stmt: mod.getOps()) {

      if (auto andOp = dyn_cast<comb::AndOp>(stmt)) {
        if (succeeded(lowerOp<comb::AndOp>(andOp))) toRemove.push_back(andOp);
      } else if (auto addOp = dyn_cast<comb::AddOp>(stmt)) {
        if (succeeded(lowerOp<comb::AddOp>(addOp))) toRemove.push_back(addOp);
      } else if (auto orOp = dyn_cast<comb::OrOp>(stmt)) {
        if (succeeded(lowerOp<comb::OrOp>(orOp))) toRemove.push_back(orOp);
      } else if (auto xorOp = dyn_cast<comb::XorOp>(stmt)) {
        if (succeeded(lowerOp<comb::XorOp>(xorOp))) toRemove.push_back(xorOp);
      } else if (auto addOp = dyn_cast<comb::AddOp>(stmt)) {
        if (succeeded(lowerOp<comb::AddOp>(addOp))) toRemove.push_back(xorOp);
      } else if (auto repOp = dyn_cast<comb::ReplicateOp>(stmt)) {
        if (succeeded(lowerReplicateOp(repOp))) toRemove.push_back(repOp);
      } else if (auto shruOp = dyn_cast<comb::ShrUOp>(stmt)) {
        if (succeeded(lowerShru_1b(shruOp))) toRemove.push_back(shruOp);
      } else if (auto constVecOp = dyn_cast<hw::AggregateConstantOp>(stmt)) {
        // if (succeeded(lowerAggregateConsts(constVecOp))) toRemove.push_back(constVecOp);
        if (failed(lowerAggregateConsts(constVecOp))) return failure();
        toRemove.push_back(constVecOp);
        for (auto eachUser: constVecOp->getUsers()) toRemove.push_back(eachUser);
      } else if (auto vecOp = dyn_cast<hw::ArrayCreateOp>(stmt)) {
        // if (succeeded(lowerArray(vecOp))) toRemove.push_back(vecOp);
        if (failed(lowerArray(vecOp))) return failure();
        toRemove.push_back(vecOp);
        for (auto eachUser: vecOp->getUsers()) toRemove.push_back(eachUser);
      } else if (
        isa<hw::ArrayConcatOp>(stmt) 
        || isa<hw::ArraySliceOp>(stmt)
        || isa<hw::StructCreateOp>(stmt)
        || isa<hw::UnionCreateOp>(stmt)
      ) {
        return stmt.emitError() << "Unsupported op " << stmt.getName();
      }

    }

    if (!toRemove.empty()) {
      LLVM_DEBUG(
        char buffer[128];
        format("Removing %d Ops\n", toRemove.size()).snprint(buffer, 128);
        llvm::dbgs() << buffer
        );
      for (auto op: llvm::reverse(toRemove)) op->erase();
    }

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

std::unique_ptr<mlir::Pass> toucan::createLowerCombPreProcessPass() {
  return std::make_unique<LowerCombPreProcessPass>();
}
