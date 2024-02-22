
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
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"



using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;


IsLegalToucan4B::IsLegalToucan4B(Operation *op) {
  isToucanOnly = true;
  is4BOnly = true;

  if (auto modOp = dyn_cast<ModuleOp>(op)) {
    auto ops = modOp.getOps();
    auto numOps = std::distance(ops.begin(), ops.end());
    auto chunkCnt = (numOps + chunkSize - 1) / chunkSize;
    
    auto ret = failableParallelForEachN(op->getContext(), 0, chunkCnt, [&](size_t chunkId){
      auto startIter = std::next(ops.begin(), chunkId * chunkSize);
      size_t visitCnt = 0;
      for (auto it = startIter; it != ops.end() && visitCnt <= chunkSize; it++, visitCnt++) {
        auto opDialect = it->getDialect();
        if (!isa<toucan::ToucanDialect>(opDialect)) {
          it->emitError() << "Is not toucan dialect! (got " << it->getName() << ")";
          isToucanOnly = false;
          return failure();
        }
        for (auto result: it->getResults()) {
          auto resultBits = hw::getBitWidth(result.getType());
          if (resultBits > 4) {
            it->emitError() <<"Result wider than 4b! (got " << resultBits << "\n";
            is4BOnly = false;
            return failure();
          }
        }
      }
      return success();
    });

    isLegalToucan4B = succeeded(ret);
  }
}


