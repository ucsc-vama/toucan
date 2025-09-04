
#pragma once

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include "circt/Dialect/HW/HWDialect.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "toucan/ToucanTypes.h"
#include "toucan/PartitioningGraph.h"
#include "toucan/MicroPartLocalValueAllocator.h"
#include "toucan/ToucanCodeGenInfo.h"
#include "toucan/ToucanConfigs.h"

#include "toucan/MicroPartitioner.h"

#include <boost/graph/adjacency_list.hpp>

#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>


namespace toucan {


  class SchedulerBase {
  public:

    // void fillDebugInfo(bool fillSignalDebugInfo = true);

    static void getVtxToLevel(const PartitioningGraph &g, mlir::SmallVector<uint32_t> &levels, uint32_t maxVtxId);
    // static void collectPrintString(const DesignGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t>  &printStrings);
    static void collectPrintString(const PartitioningGraph &graph, mlir::DenseMap<mlir::StringRef, uint32_t>  &printStrings);
    static void levelizeWorker(const PartitioningGraph &g, mlir::SmallVector<mlir::SmallVector<uint32_t>> &graphLevels);
    static void populateOpMetaDebugInfo(CGOpMetaInfo &opMeta, mlir::Operation *op);
  };

}
