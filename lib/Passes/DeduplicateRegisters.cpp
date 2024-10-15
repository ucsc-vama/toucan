
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

      for (auto eachRegWrite: sameRegWriteOps) {
        auto regValToMerge = eachRegWrite.getReg();
        
        if (eachRegWrite != op) {
          // Merge this reg with current op reg
          assert(regValToMerge != regVal);

          auto regValDefOp = regValToMerge.getDefiningOp();


          // auto toMergeRegValNamehint = getSVNameHintAttr(eachRegWrite);
          // LLVM_DEBUG(llvm::dbgs() << "Merging " << regValNamehint << " with " << toMergeRegValNamehint << "\n");

          regValToMerge.replaceAllUsesWith(regVal);

          toRemove.insert(eachRegWrite);
          toRemove.insert(regValDefOp);

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
