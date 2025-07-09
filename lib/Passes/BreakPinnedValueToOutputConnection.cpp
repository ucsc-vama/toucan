
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"

#include <cassert>
#include <memory>



#define GEN_PASS_DEF_BREAKPINNEDVALUETOOUTPUTCONNECTION
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"

#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "BreakPinnedValueToOutputConnectionPass"

struct BreakPinnedValueToOutputConnectionPass : toucan::impl::BreakPinnedValueToOutputConnectionBase<BreakPinnedValueToOutputConnectionPass> {
  using BreakPinnedValueToOutputConnectionBase<BreakPinnedValueToOutputConnectionPass>::BreakPinnedValueToOutputConnectionBase;

  static bool isPinnedEdgeSource(mlir::Operation *op) {
    return isa<toucan::RegReadOp>(op) || isa<toucan::StaticVectorSegmentReadOp>(op);
  }
  static bool isPinnedEdgeTarget(mlir::Operation *op) {
    return isa<toucan::RegWriteOp>(op) || isa<toucan::StopOp>(op) || isa<toucan::PrintOp>(op);
  }

  void runOnOperation() final {
    // mlir::DenseSet<APInt> constVals;
    // mlir::DenseSet<APInt> constVecs;
    mlir::DenseSet<mlir::Value> valuesNeedAddNOP;


    getOperation()->walk([&](toucan::RegReadOp op){
      for (const auto &user: op->getUsers()) {
        if (isPinnedEdgeTarget(user)) {
          valuesNeedAddNOP.insert(op.getResult());
          return;
        }
      }
    });
    getOperation()->walk([&](toucan::StaticVectorSegmentReadOp op){
      for (const auto &user: op->getUsers()) {
        if (isPinnedEdgeTarget(user)) {
          valuesNeedAddNOP.insert(op.getResult());
          return;
        }
      }
    });

    insertedNOPs = valuesNeedAddNOP.size();
    breakedEdges = 0;

    for (auto &eachVal: valuesNeedAddNOP) {
      OpBuilder builder(eachVal.getDefiningOp());
      // IRRewriter rewriter(builder);

      auto nopOp = builder.create<toucan::LUTOp>(eachVal.getLoc(), toucan::LUTOpName::LUT_Nop, eachVal);
      auto nopVal = nopOp.getResult();
      int nopUseCount = 0;

      // eachVal.print(dbgs());
      // dbgs() << "\nUsers:\n";

      for (auto &eachUse: eachVal.getUses()) {
        // eachUse.getOwner()->print(dbgs());
        // dbgs() << "\n";
        if (isPinnedEdgeTarget(eachUse.getOwner())) {
          // dbgs() << "Replace this\n";
          breakedEdges++;
          nopUseCount++;
          eachUse.set(nopVal);
        }
      }

      assert(nopUseCount != 0);
    }

  }
};

std::unique_ptr<mlir::Pass> toucan::createBreakPinnedValueToOutputConnectionPass() {
  return std::make_unique<BreakPinnedValueToOutputConnectionPass>();
}