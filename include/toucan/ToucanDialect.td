#ifndef TOUCAN_DIALECT
#define TOUCAN_DIALECT

include "mlir/IR/OpBase.td"

def ToucanDialect : Dialect {
  let name = "toucan";
  let cppNamespace = "::toucan";
  let summary = "IR nodes used by Toucan";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif
