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
    $reg attr-dict `:` type($reg) `->` type($result)
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
    $reg $data attr-dict `:` `(` type($reg)  type($data) `)`
  }];

}



def DefConstVectorOp : ToucanOp<"const_vector", [Pure]> {
  let summary = "def const vector";
  let description = summary;

  let arguments = (ins ArrayAttr: $values);
  let results = (outs VecType: $handle);
  let assemblyFormat = [{
    $values attr-dict `:` type($handle)
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
    $inputs attr-dict `:` type($inputs) `->` type($handle)
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

def StaticVectorSegmentReadOp : ToucanOp<"vec_read_segment", [Pure]> {
  let summary = "Statically read a single 4b segment of a vector. This op should NOT generate in final bitstream. It only works as a hint to value allocator";
  let description = summary;

  let hasVerifier = 1;

  let arguments = (ins
    VecType: $handle,
    APIntAttr: $segment_id
  );
  let results = (outs HWIntegerType: $result);

  let builders = [
    OpBuilder<(ins "mlir::Value": $vec, "int16_t": $segment), [{
      auto resultType = $_builder.getIntegerType(4);
      auto segmentIdAttr = $_builder.getI16IntegerAttr(segment);
      return build($_builder, $_state, resultType, vec, segmentIdAttr);
    }]>
  ];

  let assemblyFormat = "$handle `segment` $segment_id attr-dict `:` type($handle) `->` type($result)";
}

def VectorLogicOp : ToucanOp<"vec_logic", [Pure]> {
  let summary = "vec logical operations";
  let description = summary;

  let arguments = (ins
    VecLogicOpNameAttr: $opName,
    VecType: $v1,
    VecType: $v2
  );
  let results = (outs HWIntegerType: $result);

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "toucan::VecLogicOpName": $opName, "mlir::Value": $v1, "mlir::Value": $v2), [{
      // inputs should be a vector
      auto resultType = $_builder.getIntegerType(1);
      build($_builder, $_state, resultType, opName, v1, v2);
    }]>,
  ];

  let assemblyFormat = "$opName $v1 $v2 attr-dict `:` type($v1) type($v2) `->` type($result)";
}

def VectorArithOp : ToucanOp<"vec_arith", [Pure]> {
  let summary = "vec arithmetic operations";
  let description = summary;

  let arguments = (ins
    VecArithOpNameAttr: $opName,
    VecType: $v1,
    VecType: $v2
  );
  let results = (outs VecType: $result);

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "toucan::VecArithOpName": $opName, "mlir::Value": $v1, "mlir::Value": $v2), [{
      auto resultVecLength = v1.getType().cast<VecType>().getLength();
      auto resultVecElemType = v1.getType().cast<VecType>().getElementType();
      auto vecType = $_builder.getAttr<toucan::VecType>(false, resultVecLength, resultVecElemType);
      return build($_builder, $_state, vecType, opName, v1, v2);
    }]>,
  ];

  let assemblyFormat = "$opName $v1 $v2 attr-dict `:` type($v1) type($v2) `->` type($result)";
}

def MemReadOp: ToucanOp<"mem_read", [Pure]> {
  let summary = "memory read";
  let description = summary;

  let arguments = (ins
    MemType: $mem,
    // HWIntegerType: $addr,
    // Variadic<HWIntegerType>: $addrs,
    VecType: $addrVec,
    I1: $en
  );

  let results = (outs AnySignlessInteger: $result);
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$mem, "mlir::Value":$addrVec, "mlir::Value":$en), [{
      auto dataType = mem.getType().cast<toucan::MemType>().getElementType();
      build($_builder, $_state, dataType, mem, addrVec, en);
    }]>
  ];

  let assemblyFormat = "$mem `[` type($addrVec) $addrVec `]` `en` $en attr-dict `:` type($mem) `->` type($result)";

}

def MemWriteOp: ToucanOp<"mem_write"> {
  let summary = "memory write";
  let description = summary;
  let hasCanonicalizeMethod = 1;

  let arguments = (ins 
    MemType: $mem,
    // HWIntegerType: $addr,
    // Variadic<HWIntegerType>: $addrs,
    VecType: $addrVec,
    HWIntegerType: $data,
    I1: $en
  );
  let hasVerifier = 1;

  let assemblyFormat = "$mem `[` $addrVec `]` `=` $data `en` $en attr-dict `:` type($mem) `[` type($addrVec) `]` type($data) `en` type($en) `->` `(``)`";
}

def DefMemOp : ToucanOp<"def_mem"> {
  let summary = "def memory";
  let description = summary;
  // let arguments = ();

  let results = (outs MemType: $handle);

  let assemblyFormat = [{
    attr-dict `:` type($handle)
  }];

}

def DefRegOp : ToucanOp<"def_reg"> {
  let summary = "def register";
  let description = summary;
  // let arguments = (ins SymbolNameAttr:$sym_name);

  let results = (outs RegType: $handle);

  let builders = [
    OpBuilder<(ins "mlir::IntegerType":$element_type), [{
      build($_builder, $_state, $_builder.getType<RegType>(element_type));
    }]>
  ];

  let assemblyFormat = [{
    attr-dict `:` type($handle)
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
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = "$op_name $inputs attr-dict `:` `(` type($inputs) `)` `->` type($result)";

  let extraClassDeclaration = [{
    static size_t getResultWidth1(toucan::LUTOpName opName, mlir::ValueRange inputs);
    static size_t getResultWidth2(toucan::LUTOpName opName, mlir::ValueRange inputs);
    static size_t getResultWidth3(toucan::LUTOpName opName, mlir::ValueRange inputs);
    static size_t getLegalOperandCount(toucan::LUTOpName opName);
    static toucan::LUTOpName getMappedOpName(toucan::LUTOpName opName);
  }];

  let builders = [
    OpBuilder<(ins "toucan::LUTOpName":$op_enum, "mlir::Value":$inputVal), [{
      auto inputs = ValueRange({inputVal});
      auto resultWidth = getResultWidth1(op_enum, inputVal);
      auto resultType = $_builder.getIntegerType(resultWidth);
      auto realOpName = getMappedOpName(op_enum);
      build($_builder, $_state, resultType, realOpName, inputs);
    }]>,
    OpBuilder<(ins "toucan::LUTOpName":$op_enum, "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      auto inputs = ValueRange({lhs, rhs});
      auto resultWidth = getResultWidth2(op_enum, inputs);
      auto resultType = $_builder.getIntegerType(resultWidth);
      auto realOpName = getMappedOpName(op_enum);
      build($_builder, $_state, resultType, realOpName, inputs);
    }]>,
    OpBuilder<(ins "toucan::LUTOpName":$op_enum, "mlir::Value":$op1, "mlir::Value":$op2, "mlir::Value":$op3), [{
      auto inputs = ValueRange({op1, op2, op3});
      auto resultWidth = getResultWidth3(op_enum, inputs);
      auto resultType = $_builder.getIntegerType(resultWidth);
      auto realOpName = getMappedOpName(op_enum);
      build($_builder, $_state, resultType, realOpName, inputs);
    }]>
  ];

}


def ConstantOp : ToucanOp<"const", [Pure]> {
  let summary = "constant";
  let description = summary;
  
  let arguments = (ins APIntAttr:$value);
  let results = (outs HWIntegerType:$result);
  
  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];


  let builders = [
    /// Build a ConstantOp from an APInt, infering the result type from the
    /// width of the APInt.
    OpBuilder<(ins "const mlir::APInt &":$value)>,

    /// This builder allows construction of small signed integers like 0, 1, -1
    /// matching a specified MLIR IntegerType.  This shouldn't be used for
    /// general constant folding because it only works with values that can be
    /// expressed in an int64_t.  Use APInt's instead.
    OpBuilder<(ins "mlir::Type":$type, "int64_t":$value)>,

    /// Build a ConstantOp from a prebuilt attribute.
    OpBuilder<(ins "mlir::IntegerAttr":$value)>
  ];
}


#endif
