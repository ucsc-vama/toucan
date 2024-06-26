
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>


#define GEN_PASS_DEF_GPUCODEGEN
#include "toucan/ToucanPassCommon.h"

#include "toucan/CodeGenCommon.h"
#include "ToucanSim/ToucanGenDataTypes.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "GPUCodeGenPass"

struct GPUCodeGenPass : toucan::impl::GPUCodeGenBase<GPUCodeGenPass>, CodeGenHelper {
  using GPUCodeGenBase<GPUCodeGenPass>::GPUCodeGenBase;


  void runOnOperation() final {
    // Mark all analyses as preserved. This is a read only pass
    markAllAnalysesPreserved();

    auto graph = getAnalysis<DesignGraph>();

    auto p = RepCutPartitioner(outputDirectory.getValue());

    auto result = p.partitionAndSchedule(&getContext(), graph);

    if (failed(result)) {
      return signalPassFailure();
    }

  }

};

std::unique_ptr<mlir::Pass> toucan::createGPUCodeGenPass(GPUCodeGenOptions option) {
  return std::make_unique<GPUCodeGenPass>(option);
}