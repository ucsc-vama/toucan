#ifndef TOUCAN_PASSES_TD
#define TOUCAN_PASSES_TD

include "mlir/Pass/PassBase.td"


def RemoveSVnOM: Pass<"toucan-remove-sv-om", "circt::ModuleOp"> {
  let summary = "Remove SV and OM dialect ops";
  let description = summary;
  let constructor = "toucan::createRemoveSVnOMPass()";
  let statistics = [
    Statistic<"removedOps", "RemovedOps", "Operations removed">
  ];
}

def ExpandSVMacro: Pass<"expand-sv-macro", "circt::ModuleOp"> {
  let summary = "Expand SV macro defs.";
  let description = "Expand SystemVerilog macros; Replace sv.fwrite and sv.fatal with toucan.print and toucan.stop. Note this pass only supports partial of SV dialect that generates by FIRRTL frontend";
  let constructor = "toucan::createExpandSVMacroPass()";
  let statistics = [
    Statistic<"removedOps", "RemovedOps", "Operations removed">,
    Statistic<"printOps", "PrintOps", "SV.FWriteOp encountered (translated to toucan.print)">,
    Statistic<"stopOps", "StopOps", "SV.FatalOp encountered (translated to toucan.stop)">
  ];
}

def FactorArrayGetMux: Pass<"factor-array-get-mux", "circt::ModuleOp"> {
  let summary = "Remove Mux before array_get";
  let description = summary;
  let constructor = "toucan::createFactorArrayGetMuxPass()";
  let statistics = [
    Statistic<"arrayMuxLowered", "ArrayMuxLowered", "Number of muxes that take hw.array as inputs. They are removed at the cost of extra ArrayGetOp">
  ];
}

def SplitFirMemRWPorts: Pass<"split-firmem-rw-ports", "circt::ModuleOp"> {
  let summary = "Split FirMem ReadWritePort into ReadPort and WritePort";
  let description = summary;
  let constructor = "toucan::createSplitFirMemRWPortsPass()";
  let statistics = [
    Statistic<"splittedRWPorts", "SplittedRWPorts", "Number of seq::FirMemReadWriteOp splitted">
  ];
}

def ExpandMemoryDelay: Pass<"expand-memory-delay", "circt::ModuleOp"> {
  let summary = "Breaking firmem with read latency > 0 into combinational-read memories and pipeline registers";
  let description = summary;
  let constructor = "toucan::createExpandMemoryDelayPass()";
  let statistics = [
    Statistic<"nonZeroReadLatencyMems", "Non-zero read latency mem", "Number of memories that has read latency > 0">,
    Statistic<"nonOneWriteLatencyMems", "Non-one write latency mem", "Number of memories that has write latency != 1">,
    Statistic<"insertedRegs", "InsertedRegisters", "Number of pipelining registers inserted">,
    Statistic<"noNameMems", "AnonymousMems", "Number of memories with no name">
  ];
}

def ReplaceAsyncResetRegs: Pass<"replace-async-reset-regs", "circt::ModuleOp"> {
  let summary = "Replace async reset registers with sync reset";
  let description = summary;
  let constructor = "toucan::createReplaceAsyncResetRegsPass()";
  let statistics = [
    Statistic<"asyncRegs", "AsyncResetRegs", "Number of async reset registers">
  ];
}

def SplitRegisters: Pass<"split-registers", "circt::ModuleOp"> {
  let summary = "Split register read and write";
  let description = "Replace a seq::FirRegOp with toucan::DefRegOp, toucan::RegReadOp, and toucan::RegWriteOp. Reset will be replaced by a Mux";
  let constructor = "toucan::createSplitRegistersPass()";
  let statistics = [
    Statistic<"numRegs", "Registers", "Number of registers">,
    Statistic<"numResetRegs", "Registers with reset", "Number of registers with reset signal">
  ];
}

def RemoveMemMask: Pass<"remove-mem-mask", "circt::ModuleOp"> {
  let summary = "Remove memory write mask signals";
  let description = "Replace seq::FirMem with DefMem, MemWrite and MemRead. Remove write mask by splitting memories";
  let constructor = "toucan::createRemoveMemMaskPass()";
  let statistics = [
    Statistic<"numMemWithMask", "Mems with mask", "Number of memory has mask signal">,
    Statistic<"numNewMem", "New mems", "Number of new memories after splitting mask">
  ];
}


def Canonicalizer: Pass<"canonicalizer", "circt::ModuleOp"> {
  let summary = "Canonicalizer pass, parallel execution module by module";
  let description = summary;
  let constructor = "toucan::createCanonicalizerPass()";
}



def LowerRegMemTo4B: Pass<"lower-reg-mem-to-4B", "circt::ModuleOp"> {
  let summary = "Lower toucan::Mem and toucan::Reg to at most 4 bit width.";
  let description = summary;
  let constructor = "toucan::createLowerRegMemTo4BPass()";
  let statistics = [
    Statistic<"numRegsBefore", "Registers before split", "Number of registers before split to 4B">,
    Statistic<"numRegsAfter", "Registers after split", "Number of registers after split to 4B">,
    Statistic<"numMemsBefore", "Memories before split", "Number of memories before split to 4B">,
    Statistic<"numMemsAfter", "Memories after split", "Number of memories after split to 4B">
  ];
}

def EnsureNoClockOp: Pass<"ensure-no-clock", "circt::ModuleOp"> {
  let summary = "Emit error if the clock signal is being used";
  let description = "Emit error if found any seq::clock_div, seq::clock_mux, seq::clock_gate, seq::from_clock, seq::to_clock, seq::const_clock.";
  let constructor = "toucan::createEnsureNoClockOpPass()";
}


def LowerCombPreProcess: Pass<"lower-comb-pre-process", "circt::ModuleOp"> {
  let summary = "Lower comb,and, comb.add, comb.xor, comb.xor to binary op (it may contains more than 2 operands). Handle replicateOp when input width greater than 1. ";
  let description = summary;
  let constructor = "toucan::createLowerCombPreProcessPass()";
  let statistics = [
    Statistic<"numLongRep", "Long Replicate Op", "Number of long replicate (input width > 1)">,
    Statistic<"numMultiOprandBinaryOp", "Multi-Oprand Binary Op", "Number of binary op (Add, And, Or, Xor) with more than 2 oprand">,
    Statistic<"num1bShru", "1 Bit ShrU", "Number of 1 bit ShrU op">,
    Statistic<"numConstArray", "hw::AggregateConstOp", "Number of hw::AggregateConstOp">,
    Statistic<"numHWArray", "hw::ArrayCreateOp", "Number of hw::ArrayCreateOp">,
    Statistic<"numSmallHWArray", "hw::ArrayCreateOp small", "Number of hw::ArrayCreateOp that is no longer than 8 elements. Will lower to mux chain">
  ];
}

def LowerCombTo4B_ShortReplicateOp: Pass<"lower-comb-to-4b-short-replicateop", "circt::ModuleOp"> {
  let summary = "Lower comb::ReplicateOp with 1-bit input to 4 bits";
  let description = "Why a seperate pass? Because the canonicalizer may create ReplicateOp.";
  let constructor = "toucan::createLowerCombTo4B_ShortReplicateOpPass()";
  let statistics = [
    Statistic<"numShortRepOp", "Short Replicate Op", "Number of short replicate (input width == 1)">
  ];
}

def LowerCombTo4B_1: Pass<"lower-comb-to-4b-1", "circt::ModuleOp"> {
  let summary = "Lower comb to 4 bits, stage 1";
  let description = summary;
  let constructor = "toucan::createLowerCombTo4B_1Pass()";
  let statistics = [
    Statistic<"numCombShl", "comb::ShlOp", "Number of comb::ShlOp">,
    Statistic<"numCombShrU", "comb::ShrUOp", "Number of comb::ShrUOp">,
    Statistic<"numCombShrS", "comb::ShrSOp", "Number of comb::ShrSOp">,
    Statistic<"numCombICmp", "comb::ICmpOp", "Number of comb::ICmpOp">,
    Statistic<"numCombMul", "comb::MulOp", "Number of comb::MulOp">,
    Statistic<"numCombParity", "comb::ParityOp", "Number of comb::ParityOp">,
    Statistic<"numShiftToArray", "Shift to array", "Num of shifts lowered to array. 1 shift, 1 array, multiple read">,
    Statistic<"numArrayReadFromShift", "Array read from shift", "Num of array read op lowered from shifts">
  ];
}

def LowerCombTo4B_2: Pass<"lower-comb-to-4b-2", "circt::ModuleOp"> {
  let summary = "Lower comb to 4 bits, stage 2";
  let description = summary;
  let constructor = "toucan::createLowerCombTo4B_2Pass()";
  let statistics = [
    Statistic<"numCombAdd", "comb::AddOp", "Number of comb::AddOp">,
    Statistic<"numCombSub", "comb::SubOp", "Number of comb::SubOp">,
    Statistic<"numCombMux", "comb::MuxOp", "Number of comb::MuxOp">,
  ];
}

def LowerCombTo4B_3: Pass<"lower-comb-to-4b-3", "circt::ModuleOp"> {
  let summary = "Lower comb to 4 bits, stage 3";
  let description = summary;
  let constructor = "toucan::createLowerCombTo4B_3Pass()";
  let statistics = [
    Statistic<"numCombAnd", "comb::AndOp", "Number of comb::AndOp">,
    Statistic<"numCombOr", "comb::OrOp", "Number of comb::OrOp">,
    Statistic<"numCombXor", "comb::XorOp", "Number of comb::XorOp">,
  ];
}

def Flatten: Pass<"flatten", "circt::ModuleOp"> {
  let summary = "Flatten whole design, include top module";
  let description = summary;
  let constructor = "toucan::createFlattenPass()";
  let statistics = [
    Statistic<"numEdgesBeforeFlatten", "Before flatten, num of edges", "number of edges before flatten">,
    Statistic<"numOpsBeforeFlatten", "Before flatten, num of ops", "number of ops before flatten">,
    Statistic<"numEdgesAfterFlatten", "After flatten, num of edges", "number of edges after flatten">,
    Statistic<"numOpsAfterFlatten", "After flatten, num of ops", "number of ops after flatten">
  ];
}

def FactorConcatExtract: Pass<"factor-concat-extract", "circt::ModuleOp"> {
  let summary = "Replace comb::Extract and comb::Concat. This pass should finally remove all of them.";
  let description = summary;
  let constructor = "toucan::createFactorConcatExtractPass()";
}

def FlatDCE: Pass<"flat-dce", "circt::ModuleOp"> {
  let summary = "DCE pass on flatten design";
  let description = summary;
  let constructor = "toucan::createFlatDCEPass()";

  let statistics = [
    Statistic<"removedOps", "removedOps", "Operations removed by DCE pass">
  ];
}

def EnsureToucanOnly: Pass<"ensure-toucan-only", "circt::ModuleOp"> {
  let summary = "Ensure output only contains toucan dialect, and all values are within 4B";
  let description = summary;
  let constructor = "toucan::createEnsureToucanOnlyPass()";
}

#endif
