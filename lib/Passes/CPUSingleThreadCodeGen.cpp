
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
#include "toucan/ToucanAnalysis.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"

#include <memory>



#define GEN_PASS_DEF_CPUSINGLETHREADCODEGEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/CodeGenCommon.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "CPUSingleThreadCodeGenPass"

struct CPUSingleThreadCodeGenPass : toucan::impl::CPUSingleThreadCodeGenBase<CPUSingleThreadCodeGenPass>, CodeGenHelper {
  using CPUSingleThreadCodeGenBase<CPUSingleThreadCodeGenPass>::CPUSingleThreadCodeGenBase;


  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto partitionResult = getAnalysis<NaivePartitioner>();
    populateLUT();
    

  }

};

std::unique_ptr<mlir::Pass> toucan::createCPUSingleThreadCodeGenPass(CPUSingleThreadCodeGenOptions option) {
  return std::make_unique<CPUSingleThreadCodeGenPass>(option);
}