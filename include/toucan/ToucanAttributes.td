#ifndef TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD
#define TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD

include "toucan/ToucanDialect.td"
include "mlir/IR/EnumAttr.td"

let cppNamespace = "toucan" in {

    def LUTOpNameAttr: I32EnumAttr<"LUTOpName", "lookup table operations", [
        I32EnumAttrCase<"LUT_Nop", 0, "nop">,
        I32EnumAttrCase<"LUT_And", 1, "and">,
        I32EnumAttrCase<"LUT_Or", 2, "or">,
        I32EnumAttrCase<"LUT_Xor", 3, "xor">,
        I32EnumAttrCase<"LUT_Rep1b", 4, "rep1b">,
        I32EnumAttrCase<"LUT_Shl1", 5, "shl1">,
        I32EnumAttrCase<"LUT_Shl2", 6, "shl2">,
        I32EnumAttrCase<"LUT_Shl3", 7, "shl3">,
        I32EnumAttrCase<"LUT_Shr1", 8, "shr1">,
        I32EnumAttrCase<"LUT_Shr2", 9, "shr2">,
        I32EnumAttrCase<"LUT_Shr3", 10, "shr3">,
        I32EnumAttrCase<"LUT_Cmp_Eq", 11, "cmp_eq">
    ]> ;

}
#endif
