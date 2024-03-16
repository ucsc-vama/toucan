#ifndef TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD
#define TOUCAN_DIALECT_TOUCAN_ATTRIBUTES_TD

include "toucan/ToucanDialect.td"
include "mlir/IR/EnumAttr.td"

let cppNamespace = "toucan" in {

    def LUTOpNameAttr: I32EnumAttr<"LUTOpName", "lookup table operations", [
        I32EnumAttrCase<"LUT_Nop",    0, "nop">,
        I32EnumAttrCase<"LUT_And",    1, "and">,
        I32EnumAttrCase<"LUT_Or",     2, "or">,
        I32EnumAttrCase<"LUT_Xor",    3, "xor">,
        I32EnumAttrCase<"LUT_Rep1b",  4, "rep1b">,
        I32EnumAttrCase<"LUT_Cmp_Eq", 5, "cmp_eq">,
        I32EnumAttrCase<"LUT_Mul_Hi", 6, "mul_hi">,
        I32EnumAttrCase<"LUT_Mul_Lo", 7, "mul_lo">,
        I32EnumAttrCase<"LUT_Add",    8, "add">,
        I32EnumAttrCase<"LUT_Carry",  9, "carry">,
        I32EnumAttrCase<"LUT_Mux",    10, "mux">,
        I32EnumAttrCase<"LUT_DShl",   11, "dshl">,
        I32EnumAttrCase<"LUT_DShr",   12, "dshr">,
        I32EnumAttrCase<"LUT_XorR",   13, "xorr">,
        I32EnumAttrCase<"LUT_Shl1",   14, "shl1">,
        I32EnumAttrCase<"LUT_Shl2",   15, "shl2">,
        I32EnumAttrCase<"LUT_Shl3",   16, "shl3">,
        // Shr ops are just placeholder. They have same truth table as shl ops
        I32EnumAttrCase<"LUT_Shr1",   17, "shr1">,
        I32EnumAttrCase<"LUT_Shr2",   18, "shr2">,
        I32EnumAttrCase<"LUT_Shr3",   19, "shr3">,
    ]> ;

}
#endif
