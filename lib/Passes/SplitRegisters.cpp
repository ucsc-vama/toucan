
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

#include <memory>
#include <atomic>


#define GEN_PASS_DEF_SPLITREGISTERS
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "SplitRegistersPass"

static std::atomic<uint64_t> numRegsInModule;
static std::atomic<uint64_t> numResetRegsInModule;

struct SplitRegistersPass : toucan::impl::SplitRegistersBase<SplitRegistersPass> {
  using SplitRegistersBase<SplitRegistersPass>::SplitRegistersBase;

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    for (auto &stmt: mod.getOps()) {
      if (auto regOp = dyn_cast<seq::FirRegOp>(stmt)) {
        assert(regOp.getIsAsync() == false && "Async reset registers should not appear here!");

        OpBuilder builder(regOp);
        IRRewriter rewriter(builder);
        rewriter.setInsertionPointAfter(regOp);

        auto regName = getSVNameHintAttr(regOp).value_or(regOp.getNameAttr());

        // auto regName = regOp.getNameAttr();
        auto elemType = cast<mlir::IntegerType>(regOp.getData().getType());
        // auto regWidth = hw::getBitWidth(elemType);

        // Declare register
        auto regDefOp = rewriter.create<toucan::DefRegOp>(regOp.getLoc(), elemType);
        auto regDefReference = regDefOp.getHandle();

        setSVNameHintAttr(regDefOp, regName);

        // Read 
        auto regReadOp = rewriter.create<toucan::RegReadOp>(regOp.getLoc(), regDefReference);
        // Set namehint for future use
        setSVNameHintAttr(regReadOp, regName);
        // Replace reads
        rewriter.replaceAllUsesWith(regOp.getResult(), regReadOp.getResult());

        // Factor reg write
        auto nextValue = regOp.getNext();
        auto nextValueName = rewriter.getStringAttr(regName + getRegNextSuffix());

        if (regOp.hasReset()) {
          auto resetValue = regOp.getResetValue();
          auto resetSignal = regOp.getReset();

          // Reset mux, choose value to appear in next cycle base on reset signal
          auto resetMux = rewriter.create<comb::MuxOp>(regOp.getLoc(), resetSignal, resetValue, nextValue);
          setSVNameHintAttr(resetMux, nextValueName);

          // Reg write op
          auto regWriteOp = rewriter.create<toucan::RegWriteOp>(regOp.getLoc(), resetMux.getResult(), regDefReference);
          setSVNameHintAttr(regWriteOp, nextValueName);
          numResetRegsInModule++;
        } else {
          // This register don't have reset signal
          // Simply write next
          
          auto regWriteOp = rewriter.create<toucan::RegWriteOp>(regOp.getLoc(), nextValue, regDefReference);
          setSVNameHintAttr(regWriteOp, nextValueName);
        }
        toRemove.push_back(regOp);
        numRegsInModule++;
      }
    }

    for (auto op: toRemove) op->erase();

    return success();
  }

  void runOnOperation() final {
    auto mod = getOperation();
    numRegsInModule = 0;
    numResetRegsInModule = 0;

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

    numRegs = numRegsInModule;
    numResetRegs = numResetRegsInModule;
  }

};

std::unique_ptr<mlir::Pass> toucan::createSplitRegistersPass() {
  return std::make_unique<SplitRegistersPass>();
}
