
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
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
#include "llvm/Support/Format.h"

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
      } else if (auto repOp = dyn_cast<comb::ReplicateOp>(stmt)) {
        if (succeeded(lowerReplicateOp(repOp))) toRemove.push_back(repOp);
      }

    }

    if (!toRemove.empty()) {
      LLVM_DEBUG(
        char buffer[128];
        format("Removing %d Ops\n", toRemove.size()).snprint(buffer, 128);
        llvm::dbgs() << buffer
        );
      for (auto op: toRemove) op->erase();
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
