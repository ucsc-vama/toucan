#pragma once

#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "toucan/ToucanDialect.h"

#include "llvm/Support/MathExtras.h"

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanTypes.h"

// #include "toucan/ToucanOps.h.inc"

#define GET_OP_CLASSES
#include "toucan/Toucan.h.inc"
