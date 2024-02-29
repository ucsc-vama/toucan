#ifndef TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD
#define TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD

include "toucan/ToucanDialect.td"
include "mlir/IR/EnumAttr.td"

let cppNamespace = "toucan" in {

    def LUTOpNameAttr: I32EnumAttr<"LUTOpName", "lookup table operations", [
        I32EnumAttrCase<"LUT_And",    0, "and">,
        I32EnumAttrCase<"LUT_Or",     1, "or">,
        I32EnumAttrCase<"LUT_Xor",    2, "xor">,
        I32EnumAttrCase<"LUT_Rep1b",  3, "rep1b">,
        I32EnumAttrCase<"LUT_Shl1",   4, "shl1">,
        I32EnumAttrCase<"LUT_Shl2",   5, "shl2">,
        I32EnumAttrCase<"LUT_Shl3",   6, "shl3">,
        I32EnumAttrCase<"LUT_Shr1",   7, "shr1">,
        I32EnumAttrCase<"LUT_Shr2",   8, "shr2">,
        I32EnumAttrCase<"LUT_Shr3",   9, "shr3">,
        I32EnumAttrCase<"LUT_Cmp_Eq", 10, "cmp_eq">,
        I32EnumAttrCase<"LUT_Mul_Hi", 11, "mul_hi">,
        I32EnumAttrCase<"LUT_Mul_Lo", 12, "mul_lo">,
        I32EnumAttrCase<"LUT_Add",    13, "add">,
        I32EnumAttrCase<"LUT_Carry",  14, "carry">,
        I32EnumAttrCase<"LUT_Mux",    15, "mux">,
        I32EnumAttrCase<"LUT_DShl",   16, "dshl">,
        I32EnumAttrCase<"LUT_DShr",   17, "dshr">,
        I32EnumAttrCase<"LUT_XorR",   18, "xorr">,
    ]> ;

}
#endif
