
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
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>


#define GEN_PASS_DEF_DEDUPLICATEREGISTERS
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

#include "toucan/ToucanAnalysis.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "DeduplicateRegisters"


// TODO: dump merged register list for future use (waveform, for example)


struct DeduplicateRegisters : toucan::impl::DeduplicateRegistersBase<DeduplicateRegisters> {
  using DeduplicateRegistersBase<DeduplicateRegisters>::DeduplicateRegistersBase;



  void runOnOperation() final {

    mlir::DenseSet<Operation*> toRemove;
    uint64_t dedupCnt = 0;


    getOperation()->walk([&](toucan::RegWriteOp op){
      if (toRemove.contains(op)) return;

      auto regWriteDataVal = op.getData();
      auto regVal = op.getReg();

      mlir::SmallVector<toucan::RegWriteOp> sameRegWriteOps;
      for (auto eachUser: regWriteDataVal.getUsers()) {
        if (auto eachRegWrite = dyn_cast<toucan::RegWriteOp>(eachUser)) {
          sameRegWriteOps.push_back(eachRegWrite);
        }
      }

      assert(sameRegWriteOps.size() >= 1);
      if (sameRegWriteOps.size() == 1) {
        // No duplicated regs, no need to dedup
        return;
      }

      assert(llvm::find(sameRegWriteOps, op) != sameRegWriteOps.end());

      // auto regValNamehint = getSVNameHintAttr(op);

      // merge sameRegWriteOps -> op
      // find an op that has both reader and writer
      mlir::DenseSet<mlir::TypedValue<toucan::RegType>> regsHasReader;
      // Note: here every register has a writer
      // mlir::DenseSet<mlir::Value> regsHasWriter;
      mlir::SmallVector<mlir::TypedValue<toucan::RegType>> regsToMerge;
      for (auto eachRegWrite: sameRegWriteOps) {
        auto currentReg = eachRegWrite.getReg();
        regsToMerge.push_back(currentReg);
        for (auto eachUser: currentReg.getUsers()) {
          if (isa<toucan::RegReadOp>(eachUser)) {
            regsHasReader.insert(currentReg);
            break;
          }
        }
      }

      auto targetReg = regVal;
      bool regHasReader = !regsHasReader.empty();

      if ((!regsHasReader.empty()) && (!regsHasReader.contains(targetReg))) {
        // need a reg read, but current reg has no reader
        for (auto eachReg: regsHasReader) {
          // pick any reg
          targetReg = eachReg;
          break;
        }
      }

      mlir::Value regReadVal;
      if (regHasReader) {
        bool foundRegReadVal = false;
        for (auto eachUserOp: targetReg.getUsers()) {
          if (auto regReadOp = dyn_cast<toucan::RegReadOp>(eachUserOp)) {
            regReadVal = regReadOp.getResult();
            foundRegReadVal = true;
            break;
          }
        }
        assert(foundRegReadVal);
      }



      // merge
      for (auto eachRegWrite: sameRegWriteOps) {
        auto regValToMerge = eachRegWrite.getReg();

        if (regValToMerge != targetReg) {


          auto regValDefOp = regValToMerge.getDefiningOp();
          toRemove.insert(regValDefOp);

          for (auto eachRegUserOp: regValToMerge.getUsers()) {
            if (auto regWriteOp = dyn_cast<toucan::RegWriteOp>(eachRegUserOp)) {
              // a writer. simply remove
              toRemove.insert(regWriteOp);
            } else if (auto regReadOp = dyn_cast<toucan::RegReadOp>(eachRegUserOp)) {
              assert(regHasReader);
              auto readResult_old = regReadOp.getResult();
              readResult_old.replaceAllUsesWith(regReadVal);
              toRemove.insert(regReadOp);
            } else {
              assert(false);
            }
          }

          // This replacement breaks dependency of old reg values
          // And ensures no error happens when erasing ops without order (mlir::DenseSet is not expected to guarentee orders)
          regValToMerge.replaceAllUsesWith(targetReg);
          dedupCnt++;
        }
      }
    });

    for (auto op: toRemove) {
      op->erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "Dedup " << dedupCnt << " registers\n");
  }

};

std::unique_ptr<mlir::Pass> toucan::createDeduplicateRegistersPass() {
  return std::make_unique<DeduplicateRegisters>();
}
