
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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <atomic>


#define GEN_PASS_DEF_REPLACEASYNCRESETREGS
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "ReplaceAsyncResetRegsPass"

static std::atomic<uint64_t> asyncRegsInModule;

struct ReplaceAsyncResetRegsPass : toucan::impl::ReplaceAsyncResetRegsBase<ReplaceAsyncResetRegsPass> {
  using ReplaceAsyncResetRegsBase<ReplaceAsyncResetRegsPass>::ReplaceAsyncResetRegsBase;

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    for (auto &stmt: mod.getOps()) {
      if (auto regOp = dyn_cast<seq::FirRegOp>(stmt)) {
        auto isAsyncReset = regOp.getIsAsync();
        if (isAsyncReset) {

          OpBuilder builder(regOp);
          builder.setInsertionPointAfter(regOp);
          IRRewriter rewriter(builder);

          auto resetSignal = regOp.getReset();
          auto regResetValue = regOp.getResetValue();
          auto outputMux = rewriter.create<comb::MuxOp>(regOp.getLoc(), resetSignal, regResetValue, regOp);
          rewriter.replaceAllUsesExcept(regOp, outputMux, outputMux);
          regOp.removeIsAsyncAttr();
          asyncRegsInModule++;
        }
      }
    }
    return success();
  }

  void runOnOperation() final {
    auto mod = getOperation();
    asyncRegsInModule = 0;

    SmallVector<hw::HWModuleOp> modulesToProcess;
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }

    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result)) return signalPassFailure();

    asyncRegs = asyncRegsInModule;
  }
};

std::unique_ptr<mlir::Pass> toucan::createReplaceAsyncResetRegsPass() {
  return std::make_unique<ReplaceAsyncResetRegsPass>();
}
