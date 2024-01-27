#ifndef TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD
#define TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD

include "toucan/ToucanDialect.td"
include "mlir/IR/EnumAttr.td"

let cppNamespace = "toucan" in {

    def LUTOpNameAttr: I32EnumAttr<"LUTOpName", "lookup table operations", [
        I32EnumAttrCase<"LUT_Nop", 0, "nop">,
        I32EnumAttrCase<"LUT_And", 1, "and">,
        I32EnumAttrCase<"LUT_Or", 2, "or">,
        I32EnumAttrCase<"LUT_Xor", 3, "xor">
    ]> ;

}
#endif
