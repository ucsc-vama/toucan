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
        I32EnumAttrCase<"LUT_Shl1", 7, "shl1">,
        I32EnumAttrCase<"LUT_Shl2", 8, "shl2">,
        I32EnumAttrCase<"LUT_Shl3", 9, "shl3">,
        I32EnumAttrCase<"LUT_Shr1", 10, "shr1">,
        I32EnumAttrCase<"LUT_Shr2", 11, "shr2">,
        I32EnumAttrCase<"LUT_Shr3", 12, "shr3">,
        I32EnumAttrCase<"LUT_Cmp_Eq", 13, "cmp_eq">,
        I32EnumAttrCase<"LUT_Mul_Hi", 14, "mul_hi">,
        I32EnumAttrCase<"LUT_Mul_Lo", 15, "mul_lo">,
        I32EnumAttrCase<"LUT_Add", 16, "add">,
        I32EnumAttrCase<"LUT_Carry", 17, "carry">,
        I32EnumAttrCase<"LUT_Mux", 18, "mux">,
        I32EnumAttrCase<"LUT_DShl", 19, "dshl">,
        I32EnumAttrCase<"LUT_DShr", 20, "dshr">,
        I32EnumAttrCase<"LUT_XorR", 21, "xorr">,
    ]> ;

}
#endif
