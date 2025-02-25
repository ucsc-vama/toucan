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
        I32EnumAttrCase<"LUT_Add",    5, "add">,
        I32EnumAttrCase<"LUT_Mux",    6, "mux">,
        I32EnumAttrCase<"LUT_DShl",   7, "dshl">,
        I32EnumAttrCase<"LUT_DShr",   8, "dshr">,
        I32EnumAttrCase<"LUT_XorR",   9, "xorr">,
        I32EnumAttrCase<"LUT_Shl1",   10, "shl1">,
        I32EnumAttrCase<"LUT_Shl2",   11, "shl2">,
        I32EnumAttrCase<"LUT_Shl3",   12, "shl3">,
        // Shr ops are just placeholder. They have same truth table as shl ops
        I32EnumAttrCase<"LUT_Shr1",   13, "shr1">,
        I32EnumAttrCase<"LUT_Shr2",   14, "shr2">,
        I32EnumAttrCase<"LUT_Shr3",   15, "shr3">,
    ]> ;

    def VecLogicOpNameAttr: I32EnumAttr<"VecLogicOpName", "vector logic operations", [
        I32EnumAttrCase<"VecLogic_Eq",     0, "vleq">,
        I32EnumAttrCase<"VecLogic_Lt",     1, "vllt">,
        I32EnumAttrCase<"VecLogic_Le",     2, "vlle">,
    ]>;

    def VecArithOpNameAttr: I32EnumAttr<"VecArithOpName", "vector arithmetic operations", [
        I32EnumAttrCase<"VecArith_Add",    0, "vaadd">,
        I32EnumAttrCase<"VecArith_Sub",    1, "vasub">,
        I32EnumAttrCase<"VecArith_Mul",    2, "vamul">,
        // TODO: VecArith_Mul and VecArith_Div
    ]>;

}
#endif
