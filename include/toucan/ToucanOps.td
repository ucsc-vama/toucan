#ifndef TOUCAN_OPS
#define TOUCAN_OPS

include "toucan/ToucanDialect.td"
include "toucan/ToucanTypes.td"
include "mlir/IR/OpBase.td"

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"

class ToucanOp<string mnemonic, list<Trait> traits = []> : Op<ToucanDialect, mnemonic, traits>;

def PrintOp: ToucanOp<"print"> {
  let summary = "print operation (no exit)";
  let description = summary;

  let arguments = (ins I1:$en, StrAttr:$msg);
}

def StopOp: ToucanOp<"stop"> {
  let summary = "end simulation";
  let description = summary;

  let arguments = (ins I1:$en);
}


def RegReadOp: ToucanOp<"reg_read", [Pure]> {
  let summary = "register read";
  let description = summary;

  let arguments = ( ins
    RegType: $reg 
  );

  let results = (outs AnyType: $result);
}

def RegWriteOp: ToucanOp<"reg_write"> {
  let summary = "register write";
  let description = summary;

  // Do we need an EN signal here?
  let arguments = (ins 
    HWIntegerType: $data,
    RegType: $reg
  );

}




def MemReadOp: ToucanOp<"mem_read", [Pure]> {
  let summary = "memory read";
  let description = summary;

  let arguments = (ins
    MemType: $mem,
    HWIntegerType: $addr,
    I1: $en
  );

  let results = (outs AnyType: $result);
}

def MemWriteOp: ToucanOp<"mem_write"> {
  let summary = "memory write";
  let description = summary;

  let arguments = (ins 
    MemType: $mem,
    HWIntegerType: $addr,
    HWIntegerType: $data,
    I1: $en
  );
}

def DefMemOp : ToucanOp<"def_mem", [Symbol]> {
  let summary = "def memory";
  let description = summary;
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<MemType>:$type);
}

def DefRegOp : ToucanOp<"def_reg", [Symbol]> {
  let summary = "def register";
  let description = summary;
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<RegType>:$type);
}

#endif
