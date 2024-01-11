#ifndef TOUCAN_PASSES_TD
#define TOUCAN_PASSES_TD

include "mlir/Pass/PassBase.td"


def RemoveSVnOM: Pass<"toucan-remove-sv-om", "circt::ModuleOp"> {
  let summary = "Remove OM Dialect";
  let description = summary;
  let constructor = "toucan::createRemoveSVnOMPass()";
}

def ExpandSVMacro: Pass<"toucan-expand-svmacro", "circt::ModuleOp"> {
  let summary = "Expand SystemVerilog macros";
  let description = "Handles sv.macro.decl, sv.macro.def, sv.macro.ref, sv.ifdef, sv.ifdef.procedural";
  let constructor = "toucan::createExpandSVMacroPass()";
}

#endif
