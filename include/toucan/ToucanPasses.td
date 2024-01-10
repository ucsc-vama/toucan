#ifndef TOUCAN_PASSES_TD
#define TOUCAN_PASSES_TD

include "mlir/Pass/PassBase.td"


def RemoveOM: Pass<"toucan-remove-om", "circt::ModuleOp"> {
  let summary = "Remove OM Dialect";
  let description = summary;
  let constructor = "toucan::createRemoveOMPass()";
}

// def ExpandSVMacro: Pass<"toucan-expand-svmacro", "circt::ModuleOp"> {

// }

#endif
