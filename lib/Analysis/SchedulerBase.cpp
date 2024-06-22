#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "toucan/ToucanAnalysis.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/ToucanUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>

#include <boost/graph/topological_sort.hpp>
#include <cstring>
#include <iterator>
#include <array>
#include <optional>
#include <tuple>
#include <vector>


using namespace toucan;

using namespace mlir;
using namespace llvm;
using namespace circt;


void SchedulerBase::collectPrintString(DesignGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t> &printStrings) {
  uint32_t stringId = 0;

  for (uint32_t vtxId = 0; vtxId < boost::num_vertices(graph.g); vtxId++) {
    auto vtxOpName = graph.g[vtxId].toucanOpName;
    if (vtxOpName == CGToucanOPName::Print) {
      auto printOp = cast<toucan::PrintOp>(graph.g[vtxId].op);

      auto printStr = printOp.getMsg();
      if (!printStrings.contains(printStr)) {
        // a new string
        printStrings[printStr] = stringId;
        stringId++;
      }
    }
  }
}
