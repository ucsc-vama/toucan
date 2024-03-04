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
        I32EnumAttrCase<"LUT_Cmp_Eq", 4, "cmp_eq">,
        I32EnumAttrCase<"LUT_Mul_Hi", 5, "mul_hi">,
        I32EnumAttrCase<"LUT_Mul_Lo", 6, "mul_lo">,
        I32EnumAttrCase<"LUT_Add",    7, "add">,
        I32EnumAttrCase<"LUT_Carry",  8, "carry">,
        I32EnumAttrCase<"LUT_Mux",    9, "mux">,
        I32EnumAttrCase<"LUT_DShl",   10, "dshl">,
        I32EnumAttrCase<"LUT_DShr",   11, "dshr">,
        I32EnumAttrCase<"LUT_XorR",   12, "xorr">,
        I32EnumAttrCase<"LUT_Shl1",   13, "shl1">,
        I32EnumAttrCase<"LUT_Shl2",   14, "shl2">,
        I32EnumAttrCase<"LUT_Shl3",   15, "shl3">,
        // Shr ops are just placeholder. They have same truth table as shl ops
        I32EnumAttrCase<"LUT_Shr1",   16, "shr1">,
        I32EnumAttrCase<"LUT_Shr2",   17, "shr2">,
        I32EnumAttrCase<"LUT_Shr3",   18, "shr3">,
    ]> ;

}
#endif
