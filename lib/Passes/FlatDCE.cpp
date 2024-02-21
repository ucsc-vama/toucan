
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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



#define GEN_PASS_DEF_FLATDCE
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "FlatDCEPass"

struct FlatDCEPass : toucan::impl::FlatDCEBase<FlatDCEPass> {
  using FlatDCEBase<FlatDCEPass>::FlatDCEBase;


  void runOnOperation() final {
    llvm::DenseSet<Operation*> toRemoveSet;
    llvm::SmallVector<Operation*> toRemove;

    auto saveForRemove = [&](Operation *op) {
      if (!toRemoveSet.contains(op)) {
        toRemove.push_back(op);
        toRemoveSet.insert(op);
      }
    };

    // mark unused op
    getOperation().walk([&](Operation * op) {
      if(op == getOperation()) return;
      if (op->getUses().empty()) {
        if (isPure(op)) {
          saveForRemove(op);
        }
      }
    });

    // tracing
    llvm::SmallVector<Operation*> iterResult;
    do {
      for (auto front: toRemove) {
        if (all_of(front->getUsers(), [&](Operation* use) {
          return toRemoveSet.contains(use);
          }) && isPure(front)) {
          iterResult.push_back(front);
        }
      }
      for (auto op: iterResult) {
        saveForRemove(op);
      }
      iterResult.clear();
    } while (!iterResult.empty());

    for(auto v: reverse(toRemove)) {
      v->erase();
    }

    // update statistic
    removedOps = toRemove.size();
  }

};

std::unique_ptr<mlir::Pass> toucan::createFlatDCEPass() {
  return std::make_unique<FlatDCEPass>();
}