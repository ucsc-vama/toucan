
#include "circt/Dialect/HW/HWOps.h"
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
#include "mlir/IR/ValueRange.h"
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


#define GEN_PASS_DEF_LOWERCOMBTO4B_1
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerCombTo4B_1Pass"

struct LowerCombTo4B_1Pass : toucan::impl::LowerCombTo4B_1Base<LowerCombTo4B_1Pass> {
  using LowerCombTo4B_1Base<LowerCombTo4B_1Pass>::LowerCombTo4B_1Base;

  LogicalResult splitHWConstOp(hw::ConstantOp &op) {
    SmallVector<Value> results;

    auto constValueWidth = op.getValue().getBitWidth();
    // auto constValueRaw = op.getValue().extractBits(0, 2);

    if (constValueWidth > 4) {
      OpBuilder builder(op);
      IRRewriter rewriter(builder);
      
      auto chunks = split_signal_4B(constValueWidth);
      for (auto [chunkId, chunkWidth]: chunks) {
        auto newValue = op.getValue().extractBits(chunkWidth, chunkId * 4);
        auto newConstOp = rewriter.create<hw::ConstantOp>(op->getLoc(), newValue);
        results.push_back(newConstOp.getResult());
      }

      auto bitConcatOp = rewriter.create<comb::ConcatOp>(op.getLoc(), results);
      rewriter.replaceAllUsesWith(op, bitConcatOp);
      // consider remove op
    }
    return success();
  }

  LogicalResult splitCombAndOp(comb::AndOp &op) {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    OpBuilder builder(op);
    IRRewriter rewriter(builder);

    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_And, lhs, rhs);
        intermediateResults.push_back(andOp.getResult());
      }

      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_And, lhs, rhs);
      rewriter.replaceAllUsesWith(op, andOp);
    }
    
    return success();
  }


  

  LogicalResult splitCombOrOp(comb::OrOp &op) {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());
        
    OpBuilder builder(op);
    IRRewriter rewriter(builder);
    
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Or, lhs, rhs);
        intermediateResults.push_back(andOp.getResult());
      }

      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Or, lhs, rhs);
      rewriter.replaceAllUsesWith(op, andOp);
    }
    
    return success();
  }

  LogicalResult splitCombXorOp(comb::XorOp &op) {
    if (op.getInputs().size() != 2) {
      op.emitError("Supports only 2 operands!");
    }

    auto resultValue = op.getResult();
    auto resultValueWidth = hw::getBitWidth(resultValue.getType());

    OpBuilder builder(op);
    IRRewriter rewriter(builder);
    
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    assert(hw::getBitWidth(lhs.getType()) == hw::getBitWidth(rhs.getType()));

    if (resultValueWidth > 4) {
      auto lhsValues = split_value_4B(op.getOperation(), lhs, rewriter);
      auto rhsValues = split_value_4B(op.getOperation(), rhs, rewriter);

      SmallVector<Value> intermediateResults;
      for (auto&& [lhs, rhs]: zip(lhsValues, rhsValues)) {
        auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Xor, lhs, rhs);
        intermediateResults.push_back(andOp.getResult());
      }

      concat_4b_and_replace(op.getOperation(), op.getResult(), intermediateResults, rewriter);
    } else {
      auto andOp = rewriter.create<toucan::LUTOp>(op.getLoc(), toucan::LUTOpName::LUT_Xor, lhs, rhs);
      rewriter.replaceAllUsesWith(op, andOp);
    }
    
    return success();
  }

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    for (auto &stmt: mod.getOps()) {
      // Lower hw.constant
      if (auto constOp = dyn_cast<hw::ConstantOp>(stmt)) {
        // dont do anything with const. It may be folded
        // if (failed(splitHWConstOp(constOp))) return failure();
      } else if (auto andOp = dyn_cast<comb::AndOp>(stmt)) {
        // Lower comb.and
        if (failed(splitCombAndOp(andOp))) return failure();
        toRemove.push_back(andOp);
      } else if (auto orOp = dyn_cast<comb::OrOp>(stmt)) {
        // Lower comb.or
        if (failed(splitCombOrOp(orOp))) return failure();
        toRemove.push_back(orOp);
      } else if (auto xorOp = dyn_cast<comb::XorOp>(stmt)) {
        // Lower comb.xor
        if (failed(splitCombXorOp(xorOp))) return failure();
        toRemove.push_back(xorOp);
      }

      // Lower multiplication before add

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
    // Sequential
    for (auto mod: modulesToProcess) {
      auto ret = runOnModule(mod);
      if (failed(ret)) return signalPassFailure();
    }

    // // Parallel
    // auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
    //   return runOnModule(mod);
    // });
    // if (failed(result)) return signalPassFailure();
  }

};

std::unique_ptr<mlir::Pass> toucan::createLowerCombTo4B_1Pass() {
  return std::make_unique<LowerCombTo4B_1Pass>();
}
