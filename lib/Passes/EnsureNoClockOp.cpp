
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


#define GEN_PASS_DEF_ENSURENOCLOCKOP
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "EnsureNoClockOpPass"

struct EnsureNoClockOpPass : toucan::impl::EnsureNoClockOpBase<EnsureNoClockOpPass> {
  using EnsureNoClockOpBase<EnsureNoClockOpPass>::EnsureNoClockOpBase;

  LogicalResult runOnModule(hw::HWModuleOp mod) {

    for (auto &stmt: mod.getOps()) {
      if (isa<seq::ToClockOp>(stmt)
        || isa<seq::FromClockOp>(stmt)
        || isa<seq::ConstClockOp>(stmt)
        || isa<seq::ClockGateOp>(stmt)
        || isa<seq::ClockDivider>(stmt)
        || isa<seq::ClockMuxOp>(stmt)
      ) {
        stmt.emitError() << "Clock-related operations are not supported!";
        return failure();
      }
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

std::unique_ptr<mlir::Pass> toucan::createEnsureNoClockOpPass() {
  return std::make_unique<EnsureNoClockOpPass>();
}
