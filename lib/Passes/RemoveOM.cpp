#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Support/LLVM.h"

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
#include "llvm/Support/Mutex.h"

#include <memory>



#define GEN_PASS_DEF_REMOVEOM
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct RemoveOMPass : toucan::impl::RemoveOMBase<RemoveOMPass> {
  using RemoveOMBase<RemoveOMPass>::RemoveOMBase;

  static inline bool shouldRemove(Operation * op) {
    return op->getDialect()->getNamespace() == "om";
  }

  static void removeOps(SmallVector<Operation*> &toRemove) {
    for(auto op: llvm::reverse(toRemove)) {
      // will print if -debug, or -debug-only RemoveOMPass
      DEBUG_WITH_TYPE("RemoveOMPass", llvm::dbgs() << "Removing OM Op\n");
      DEBUG_WITH_TYPE("RemoveOMPass", op->print(llvm::dbgs()));
      op->erase();
    }
  }


  static LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;

    for(auto & inner: mod.getOps()) {
      if(shouldRemove(&inner)) {
        toRemove.push_back(&inner);
      }
    }
    removeOps(toRemove);

    return success();
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<Operation*> toRemove;
    SmallVector<hw::HWModuleOp> modulesToProcess;

    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
      else if(shouldRemove(&inner)) {
        toRemove.push_back(&inner);
      }
    }

    removeOps(toRemove);

    DEBUG_WITH_TYPE("RemoveOMPass", llvm::dbgs() << "Found " << modulesToProcess.size() << " modules.\n");
    // for_each(modulesToProcess, [=](auto submod) {
    //   DEBUG_WITH_TYPE("RemoveOMPass", llvm::dbgs() << submod.getName() << "\n");
    // });

    // Search all modules. Less likely return OMOps
    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result))
        return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> toucan::createRemoveOMPass() {
  return std::make_unique<RemoveOMPass>();
}