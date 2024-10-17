
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


#define GEN_PASS_DEF_REMOVECONSTREGS
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

#include "toucan/ToucanAnalysis.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "RemoveConstRegs"


// TODO: dump merged register list for future use (waveform, for example)


struct RemoveConstRegs : toucan::impl::RemoveConstRegsBase<RemoveConstRegs> {
  using RemoveConstRegsBase<RemoveConstRegs>::RemoveConstRegsBase;



  void runOnOperation() final {

    mlir::DenseSet<Operation*> toRemove, toRemove_regDecl;

    uint64_t constRegCnt = 0;

    getOperation()->walk([&](toucan::RegWriteOp op){
      if (toRemove.contains(op)) return;

      auto regDataVal = op.getData();
      
      if (auto constantOp = dyn_cast<ConstantOp>(regDataVal.getDefiningOp())) {
        constRegCnt++;

        auto constVal = constantOp.getResult();
        auto regVal = op.getReg();

        toRemove.insert(op);
        toRemove_regDecl.insert(regVal.getDefiningOp());

        for (auto user: regVal.getUsers()) {
          if (auto regReadOp = dyn_cast<RegReadOp>(user)) {
            toRemove.insert(regReadOp);
            auto regReadResult = regReadOp.getResult();
            regReadResult.replaceAllUsesWith(constVal);
          }
        }
      }


    });

    for (auto op: toRemove) {
      op->erase();
    }
    for (auto op: toRemove_regDecl) {
      op->erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "Remove " << constRegCnt << " const registers\n");
  }

};

std::unique_ptr<mlir::Pass> toucan::createRemoveConstRegsPass() {
  return std::make_unique<RemoveConstRegs>();
}
