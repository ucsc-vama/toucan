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
  let description = "Replace a seq::FirRegOp with toucan::DefRegOp, toucan::RegReadOp, and toucan::RegWriteOp. Reset will be replaced by a Mux";
  let constructor = "toucan::createSplitRegistersPass()";
}

def RemoveMemMask: Pass<"remove-mem-mask", "circt::ModuleOp"> {
  let summary = "Remove memory write mask signals";
  let description = "Replace seq::FirMem with DefMem, MemWrite and MemRead. Remove write mask by splitting memories";
  let constructor = "toucan::createRemoveMemMaskPass()";
}


def ParallelCanonicalizer: Pass<"parallel-canonicalizer", "circt::ModuleOp"> {
  let summary = "Canonicalizer pass, parallel execution module by module";
  let description = summary;
  let constructor = "toucan::createParallelCanonicalizerPass()";
}



def LowerRegMemTo4B: Pass<"lower-reg-mem-to-4B", "circt::ModuleOp"> {
  let summary = "Lower toucan::Mem and toucan::Reg to at most 4 bit width.";
  let description = summary;
  let constructor = "toucan::createLowerRegMemTo4BPass()";
}

def EnsureNoClockOp: Pass<"ensure-no-clock", "circt::ModuleOp"> {
  let summary = "Emit error if the clock signal is being used";
  let description = "Emit error if found any seq::clock_div, seq::clock_mux, seq::clock_gate, seq::from_clock, seq::to_clock, seq::const_clock.";
  let constructor = "toucan::createEnsureNoClockOpPass()";
}


def LowerCombPreProcess: Pass<"lower-comb-pre-process", "circt::ModuleOp"> {
  let summary = "Lower comb,and, comb.add, comb.xor, comb.xor to binary op (it may contains more than 2 operands). Handle replicateOp when input width greater than 1. Warning: This pass may be reverted by canonicalizer pass!";
  let description = summary;
  let constructor = "toucan::createLowerCombPreProcessPass()";
}

def LowerCombTo4B_1: Pass<"lower-comb-to-4b-1", "circt::ModuleOp"> {
  let summary = "Lower comb to 4 bits, stage 1";
  let description = summary;
  let constructor = "toucan::createLowerCombTo4B_1Pass()";
}

def LowerCombTo4B_2: Pass<"lower-comb-to-4b-2", "circt::ModuleOp"> {
  let summary = "Lower comb to 4 bits, stage 2";
  let description = summary;
  let constructor = "toucan::createLowerCombTo4B_2Pass()";
}

def LowerCombTo4B_3: Pass<"lower-comb-to-4b-3", "circt::ModuleOp"> {
  let summary = "Lower comb to 4 bits, stage 3";
  let description = summary;
  let constructor = "toucan::createLowerCombTo4B_3Pass()";
}

def Flatten: Pass<"flatten", "circt::ModuleOp"> {
  let summary = "Flatten whole design, include top module";
  let description = summary;
  let constructor = "toucan::createFlattenPass()";
  let statistics = [
    Statistic<"numEdges", "num-edges", "number of edges after flatten">,
    Statistic<"numOps", "num-ops", "number of ops after flatten">
  ];
}

#endif
