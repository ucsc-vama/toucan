#include "toucan/ToucanTypes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "toucan/ToucanTypes.cpp.inc"

using namespace toucan;


void ToucanDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "toucan/ToucanTypes.cpp.inc"
      >();
}