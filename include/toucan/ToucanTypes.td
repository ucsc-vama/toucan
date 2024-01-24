#ifndef TOUCAN_TYPES
#define TOUCAN_TYPES

include "toucan/ToucanDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "circt/Dialect/HW/HWTypes.td"

class ToucanType<string name, list<Trait> traits=[]> : TypeDef<ToucanDialect, name, traits>;

def MemType: ToucanType<"Mem"> {
  let summary = "simulated memory type";
  let parameters = (ins
    "uint64_t": $depth,
    "mlir::IntegerType": $element_type
  );
  let mnemonic = "mem";
  let assemblyFormat = "`mem` $element_type `[` $depth `]`";
  let extraClassDeclaration = [{
    uint64_t getAddrBits() {return llvm::Log2_64_Ceil(getDepth());}
    uint64_t getElementWidth() {return getElementType().getWidth();}
  }];
}

def RegType: ToucanType<"Reg"> {
    let summary = "simulated register type";
    let parameters = (ins
        "mlir::IntegerType": $element_type
    );
    let mnemonic = "reg";
    let assemblyFormat = "`reg` $element_type";
    let extraClassDeclaration = [{
        uint64_t getElementWidth() {return getElementType().getWidth();}
    }];
}

#endif
