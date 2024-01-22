#ifndef TOUCAN_PASSES_TD
#define TOUCAN_PASSES_TD

include "mlir/Pass/PassBase.td"


def RemoveSVnOM: Pass<"toucan-remove-sv-om", "circt::ModuleOp"> {
  let summary = "Remove SV and OM dialect ops";
  let description = summary;
  let constructor = "toucan::createRemoveSVnOMPass()";
}

def FactorSV: Pass<"toucan-factor-sv", "circt::ModuleOp"> {
  let summary = "Factor SystemVerilog Ops.";
  let description = "Expand SystemVerilog macros; Replace sv.fwrite and sv.fatal with toucan.print and toucan.stop. Note this pass only supports partial of SV dialect that generates by FIRRTL frontend";
  let constructor = "toucan::createFactorSVPass()";
}

def SplitFirMemRWPorts: Pass<"split-firmem-rw-ports", "circt::ModuleOp"> {
  let summary = "Split FirMem ReadWritePort into ReadPort and WritePort";
  let description = summary;
  let constructor = "toucan::createSplitFirMemRWPortsPass()";
}

def ExpandMemoryDelay: Pass<"expand-memory-delay", "circt::ModuleOp"> {
    let summary = "Breaking firmem with read latency > 0 into combinational-read memories and pipeline registers";
    let description = summary;
    let constructor = "toucan::createExpandMemoryDelayPass()";
}

def ReplaceAsyncResetRegs: Pass<"replace-async-reset-regs", "circt::ModuleOp"> {
  let summary = "Replace async reset registers with sync reset";
  let description = summary;
  let constructor = "toucan::createReplaceAsyncResetRegsPass()";
}

def SplitRegisters: Pass<"split-registers", "circt::ModuleOp"> {
  let summary = "Split register read and write";
  let description = summary;
  let constructor = "toucan::createSplitRegistersPass()";
}

#endif
