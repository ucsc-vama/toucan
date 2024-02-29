
#pragma once

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/OM/OMDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "toucan/ToucanDialect.h"


#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"


#include <tuple>
#include <iterator>
#include <functional>

#include "toucan/ToucanAttributes.h"

namespace toucan {
  class CodeGenHelper {
    public:
    mlir::DenseMap<LUTOpName, mlir::SmallVector<uint8_t>> lutContent;

    void populateLUT();

    private:
    // 1 input gates
    void populateLUT_Rep1b();
    void populateLUT_XorR();

    // 2 inputs
    void populateLUT_And();
    void populateLUT_Or();
    void populateLUT_Xor();
    void populateLUT_Shl1();
    void populateLUT_Shl2();
    void populateLUT_Shl3();
    // void populateLUT_Shr1();
    // void populateLUT_Shr2();
    // void populateLUT_Shr3();
    void populateLUT_Cmp_Eq();
    void populateLUT_Mul_Hi();
    void populateLUT_Mul_Lo();
    void populateLUT_Carry();

    // 3 inputs
    void populateLUT_Add();
    void populateLUT_Mux();
    void populateLUT_DShl();
    void populateLUT_DShr();

  };
}

