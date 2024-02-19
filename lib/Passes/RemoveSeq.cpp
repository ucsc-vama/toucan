
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
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



#define GEN_PASS_DEF_REMOVESEQ
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "RemoveSeqPass"

struct RemoveSeqPass : toucan::impl::RemoveSeqBase<RemoveSeqPass> {
  using RemoveSeqBase<RemoveSeqPass>::RemoveSeqBase;

  static inline bool shouldRemove(Operation * op) {
    auto dialect = op->getDialect();
    return isa_and_nonnull<seq::SeqDialect>(dialect);
  }

  static void removeOps(SmallVector<Operation*> &toRemove) {
    for(auto op: llvm::reverse(toRemove)) {
      // will print if -debug, or -debug-only RemoveSeqPass
      LLVM_DEBUG(llvm::dbgs() << "Removing Seq Op\n");
      LLVM_DEBUG(llvm::dbgs() << op->getName());
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

    for(auto & inner: mod.getOps()) {
      if(shouldRemove(&inner)) {
        if (!inner.getUsers().empty()) {
          inner.emitError() << "This seq op is still used!";
          return signalPassFailure();
        }
        toRemove.push_back(&inner);
      }
    }

    removeOps(toRemove);
  }
};

std::unique_ptr<mlir::Pass> toucan::createRemoveSeqPass() {
  return std::make_unique<RemoveSeqPass>();
}