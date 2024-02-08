#ifndef TOUCAN_OPS
#define TOUCAN_OPS

include "toucan/ToucanDialect.td"
include "toucan/ToucanTypes.td"
include "toucan/ToucanAttributes.td"
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
  let hasCanonicalizeMethod = 1;

  let arguments = (ins I1:$en, StrAttr:$msg);

  let assemblyFormat = "$en $msg attr-dict";
}

def StopOp: ToucanOp<"stop"> {
  let summary = "end simulation";
  let description = summary;
  let hasCanonicalizeMethod = 1;

  let arguments = (ins I1:$en);

  let assemblyFormat = "$en attr-dict";
}


def RegReadOp: ToucanOp<"reg_read", [Pure]> {
  let summary = "register read";
  let description = summary;

  let arguments = ( ins
    RegType: $reg 
  );

  let results = (outs AnySignlessInteger: $result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$reg), [{
      build($_builder, $_state, reg.getType().cast<RegType>().getElementType(), reg);
    }]>
  ];
  
  let assemblyFormat = [{
    $reg `,` attr-dict `:` type($reg) `->` type($result)
  }];
}

def RegWriteOp: ToucanOp<"reg_write"> {
  let summary = "register write";
  let description = summary;

  // Do we need an EN signal here?
  let arguments = (ins 
    HWIntegerType: $data,
    RegType: $reg
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $reg $data `,` attr-dict `:` `(` type($reg) `,` type($data) `)`
  }];

}




def MemReadOp: ToucanOp<"mem_read", [Pure]> {
  let summary = "memory read";
  let description = summary;
  let hasCanonicalizeMethod = 1;

  let arguments = (ins
    MemType: $mem,
    HWIntegerType: $addr,
    I1: $en
  );

  let results = (outs AnySignlessInteger: $result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$mem, "mlir::Value":$addr, "mlir::Value":$en), [{
      auto dataType = mem.getType().cast<toucan::MemType>().getElementType();
      build($_builder, $_state, dataType, mem, addr, en);
    }]>
  ];

  let assemblyFormat = "$mem `[`  type($addr) $addr `]` `en` $en `,` attr-dict `:` type($mem) `->` type($result)";

}

def MemWriteOp: ToucanOp<"mem_write"> {
  let summary = "memory write";
  let description = summary;
  let hasCanonicalizeMethod = 1;

  let arguments = (ins 
    MemType: $mem,
    HWIntegerType: $addr,
    HWIntegerType: $data,
    I1: $en
  );

  let assemblyFormat = "$mem `[` $addr `]` `=` $data `en` $en attr-dict `:` type($mem) `[` type($addr) `]` type($data) `en` type($en) `->` `(``)`";
}

def DefMemOp : ToucanOp<"def_mem", [Symbol]> {
  let summary = "def memory";
  let description = summary;
  let arguments = (ins SymbolNameAttr:$sym_name);

  let results = (outs MemType: $handle);

  let assemblyFormat = [{
    $sym_name `,` attr-dict `:` type($handle)
  }];
  
  // let builders = [
  //   OpBuilder<(ins "mlir::StringRef":$sym_name, "mlir::TypeAttr":$type), [{
  //     build($_builder, $_state, $_builder.getType<MemType>(type), sym_name);
  //   }]>
  // ];
}

def DefRegOp : ToucanOp<"def_reg", [Symbol]> {
  let summary = "def register";
  let description = summary;
  let arguments = (ins SymbolNameAttr:$sym_name);

  let results = (outs RegType: $handle);

  let builders = [
    OpBuilder<(ins "mlir::StringRef":$sym_name, "mlir::IntegerType":$element_type), [{
      build($_builder, $_state, $_builder.getType<RegType>(element_type), sym_name);
    }]>
  ];

  let assemblyFormat = [{
    $sym_name `,` attr-dict `:` type($handle)
  }];

}

def LUTOp : ToucanOp<"lut", [Pure]> {
  let summary = "Toucan LUT operation";
  let description = summary;

  // A new argument indicating op type
  let arguments = (ins
    LUTOpNameAttr: $op_name,
    Variadic<HWIntegerType>: $inputs
    );

  let results = (outs HWIntegerType: $result);
  let hasVerifier = 1;

  let assemblyFormat = "$op_name $inputs `,` attr-dict `:` `(` type($inputs) `)` `->` type($result)";

  let extraClassDeclaration = [{
    static size_t getResultWidth1(toucan::LUTOpName opName, mlir::ValueRange inputs);
    static size_t getResultWidth2(toucan::LUTOpName opName, mlir::ValueRange inputs);
    static size_t getResultWidth3(toucan::LUTOpName opName, mlir::ValueRange inputs);
    static size_t getLegalOperandCount(toucan::LUTOpName opName);
  }];

  let builders = [
    OpBuilder<(ins "toucan::LUTOpName":$op_enum, "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      auto inputs = ValueRange({lhs, rhs});
      auto resultWidth = getResultWidth2(op_enum, inputs);
      auto resultType = $_builder.getIntegerType(resultWidth);
      build($_builder, $_state, resultType, op_enum, inputs);
    }]>,
    OpBuilder<(ins "toucan::LUTOpName":$op_enum, "mlir::Value":$op1, "mlir::Value":$op2, "mlir::Value":$op3), [{
      auto inputs = ValueRange({op1, op2, op3});
      auto resultWidth = getResultWidth3(op_enum, inputs);
      auto resultType = $_builder.getIntegerType(resultWidth);
      build($_builder, $_state, resultType, op_enum, inputs);
    }]>
  ];

}

def DefConstVectorOp : ToucanOp<"const_vector", [Pure]> {
  let summary = "def const vector";
  let description = summary;

  let arguments = (ins ArrayAttr: $values);
  let results = (outs VecType: $handle);
  let assemblyFormat = [{
    $values `,` attr-dict `:` type($handle)
  }];
  // let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::ArrayAttr":$values), [{
      auto elemBitWidth = hw::getBitWidth(values[0].cast<IntegerAttr>().getType());
      auto vecType = $_builder.getAttr<toucan::VecType>(true, values.size(), $_builder.getIntegerType(elemBitWidth));
      return build($_builder, $_state, vecType, values);
    }]>
  ];
}

def DefVectorOp : ToucanOp<"vector", [Pure]> {
  let summary = "def vector";
  let description = summary;

  let arguments = (ins Variadic<HWIntegerType>: $inputs);
  let results = (outs VecType: $handle);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $inputs `,` attr-dict `:` type($inputs) `->` type($handle)
  }];
  let builders = [
    OpBuilder<(ins CArg<"mlir::ArrayRef<mlir::Value>","{}">:$inputs), [{
      auto vecType = $_builder.getAttr<toucan::VecType>(false, inputs.size(), inputs[0].getType().cast<mlir::IntegerType>());
      return build($_builder, $_state, vecType, inputs);
    }]>
  ];
}

def VectorReadOp : ToucanOp<"vec_read", [Pure]> {
  let summary = "vec read";
  let description = summary;

  let arguments = (ins
    VecType: $handle,
    APIntAttr: $offset,
    HWIntegerType: $out_range_value,
    Variadic<HWIntegerType>: $indicies
  );
  let results = (outs HWIntegerType: $result);
  let builders = [
    OpBuilder<(ins "mlir::Value": $vec, "mlir::Value":$out_range_value, CArg<"mlir::ArrayRef<mlir::Value>","{}">:$indicies), [{
      auto resultType = vec.getType().cast<VecType>().getElementType();
      auto offsetAttr = $_builder.getI16IntegerAttr(0);
      return build($_builder, $_state, resultType, vec, offsetAttr, out_range_value, indicies);
    }]>,
    OpBuilder<(ins "mlir::Value": $vec, "int16_t": $offset, "mlir::Value": $out_range_value, CArg<"mlir::ArrayRef<mlir::Value>","{}">:$indicies), [{
      auto resultType = vec.getType().cast<VecType>().getElementType();
      auto offsetAttr = $_builder.getI16IntegerAttr(offset);
      return build($_builder, $_state, resultType, vec, offsetAttr, out_range_value, indicies);
    }]>
  ];

  let assemblyFormat = "$handle `[` type($indicies) $indicies `+` $offset `]` `out_range` type($out_range_value) $out_range_value attr-dict `:` type($handle) `->` type($result)";
}


// def BitScatterOp : ToucanOp<"bit_scatter"> {
//   let summary = "Split a signal into multiple signals <= 4 bits. Temporary operation.";
//   let description = summary;
//   let arguments = (ins HWIntegerType: $input);

//   let results = (outs Variadic<HWIntegerType>:$outputs);

//   let hasVerifier = 1;

//   let builders = [
//     OpBuilder<(ins "mlir::Value":$input)>
//   ];

//   let assemblyFormat = [{
//      $input `,` attr-dict `:` type($input) `->` type($outputs)
//   }];
// }

// def BitAggregateOp : ToucanOp<"bit_aggregate"> {
//   let summary = "Aggregate multiple signals <= 4 bits into 1. Temporary operation.";
//   let description = summary;
//   let arguments = (ins Variadic<HWIntegerType>: $inputs);

//   let results = (outs HWIntegerType:$output);

//   let hasVerifier = 1;

//   let builders = [
//     OpBuilder<(ins CArg<"mlir::ArrayRef<mlir::Value>","{}">:$inputs), [{
//       int64_t resultWidth = 0;
//       for (auto elem: inputs) {
//         resultWidth += hw::getBitWidth(elem.getType());
//       }
//       auto resultType = $_builder.getIntegerType(resultWidth);
//       return build($_builder, $_state, resultType, inputs);
//     }]>
//   ];

//   let assemblyFormat = [{
//     $inputs `,` attr-dict `:` type($inputs) `->` type($output)
//   }];
// }

#endif
