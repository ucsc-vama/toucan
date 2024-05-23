
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "toucan/ToucanAttributes.h"
#include "toucan/ToucanDialect.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <memory>
#include <atomic>


#define GEN_PASS_DEF_LOWERHWVECTORTO4B
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerHWVectorTo4BPass"

#define SMALL_VEC_SIZE 8

static std::atomic<uint64_t> numConstArrayInModule;
static std::atomic<uint64_t> numSmallConstArrayInModule;
static std::atomic<uint64_t> numHWArrayInModule;
static std::atomic<uint64_t> numSmallHWArrayInModule;


struct LowerConstArrayTo4B: OpRewritePattern<hw::AggregateConstantOp> {
  using OpRewritePattern<hw::AggregateConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hw::AggregateConstantOp constArrayOp, PatternRewriter &rewriter) const final {
    numConstArrayInModule++;
    
    auto zeroConstOp = rewriter.create<hw::ConstantOp>(constArrayOp.getLoc(), rewriter.getI4Type(), 0);
    auto zeroConstValue = zeroConstOp.getResult();

    SmallVector<SmallVector<Attribute>> array_values;
    SmallVector<Value> array_const_vals;
    SmallVector<Value> defVecHandles;

    auto constArrayValues = constArrayOp.getFields().getValue();

    auto elemWidth = hw::getBitWidth((constArrayValues.begin()->cast<mlir::IntegerAttr>().getType()));
    auto numChunks = (elemWidth + 3) / 4;
    for (size_t i = 0; i < static_cast<size_t>(numChunks); i++) {
      array_values.push_back({});
    }

    bool useMux = constArrayValues.size() <= SMALL_VEC_SIZE;

    if (useMux) {
      numSmallHWArrayInModule++;
      for (auto &constArrayElem: constArrayValues) {
        if (!isa<mlir::IntegerAttr>(constArrayElem)) {
          constArrayOp->emitError() << "Only supports integer arrays!";
          return failure();
        }

        auto constArrayElemVal = cast<mlir::IntegerAttr>(constArrayElem).getValue();
        auto arrayElemConstOp = rewriter.create<hw::ConstantOp>(constArrayOp.getLoc(), constArrayElemVal);
        array_const_vals.push_back(arrayElemConstOp.getResult());
      }
    } else {
      for (auto &constArrayElem: constArrayValues) {
        if (!isa<mlir::IntegerAttr>(constArrayElem)) {
          constArrayOp->emitError() << "Only supports integer arrays!";
          return failure();
        }

        auto constArrayElemVal = cast<mlir::IntegerAttr>(constArrayElem).getValue();
        auto constArrayElemValWidth = constArrayElemVal.getBitWidth();

        int startPos = (numChunks-1) * 4;
        for (auto [chunkId, chunkWidth]: (split_signal_4B(constArrayElemValWidth))) {
          assert (startPos + chunkWidth <= static_cast<int>(constArrayElemValWidth));
          auto val = constArrayElemVal.extractBits(chunkWidth, startPos);
          startPos -= 4;
          auto intAttr = rewriter.getIntegerAttr(rewriter.getIntegerType(val.getBitWidth()), val.getLimitedValue());
          // Insert by chunkId (0 for LSB), need reverse later
          array_values[chunkId].push_back(intAttr);
        }
      }

      if (array_values.size() > 1) {
        std::reverse(array_values.begin(), array_values.end());
      }

      for (auto elems: array_values) {
        auto arrayAttr = rewriter.getArrayAttr(elems);
        auto newDefVecOp = rewriter.create<toucan::DefConstVectorOp>(constArrayOp.getLoc(), arrayAttr);
        defVecHandles.push_back(newDefVecOp.getHandle());
      }
    }

    for (auto userOp: constArrayOp->getUsers()) {
      if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(userOp)) {
        rewriter.setInsertionPointAfter(arrayGetOp);

        auto arrayIndex = extractMinimumWidth(arrayGetOp.getIndex(), rewriter, userOp);
        auto arrayIndexBits = hw::getBitWidth(arrayIndex.getType());

        auto maxIndexBits = llvm::Log2_64_Ceil(constArrayValues.size());
        if (arrayIndexBits > maxIndexBits) {
          arrayGetOp.emitWarning() << "Too much index bits than necessary. (Expect index bits no more than " << maxIndexBits << ", but got " << arrayIndexBits;
        }

        if (useMux) {
          auto readResult = generate_mux_chain_for_array(arrayGetOp, rewriter, array_const_vals, arrayIndex);
          copyCustomizedAttrs(userOp, readResult.getDefiningOp());
          rewriter.replaceOp(userOp, readResult);
        } else {
          auto indicies_4b = split_value_4B(userOp, arrayIndex, rewriter);

          SmallVector<Value> arrayGetResults;
          for (auto vecHandle: defVecHandles) {
            auto readOp = rewriter.create<toucan::VectorReadOp>(arrayGetOp.getLoc(), vecHandle, zeroConstValue, indicies_4b);
            arrayGetResults.push_back(readOp);
          }

          if (arrayGetResults.size() > 1) {
            auto concatOp = rewriter.create<comb::ConcatOp>(constArrayOp.getLoc(), arrayGetResults);

            copyCustomizedAttrs(userOp, concatOp);
            rewriter.replaceOp(arrayGetOp, concatOp);
          } else {
            auto result = arrayGetResults[0];
            copyCustomizedAttrs(userOp, result.getDefiningOp());
            rewriter.replaceOp(arrayGetOp, result);
          }
        }

      } else {
        userOp->emitError() << "Unknow op using const array: " << userOp->getName();
        return failure();
      }
    }

    rewriter.eraseOp(constArrayOp);
    return success();
  }
};


struct LowerHWArrayTo4B: OpRewritePattern<hw::ArrayCreateOp> {
  using OpRewritePattern<hw::ArrayCreateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hw::ArrayCreateOp arrayCreateOp, PatternRewriter &rewriter) const final {
    numHWArrayInModule++;
    
    auto zeroConstOp = rewriter.create<hw::ConstantOp>(arrayCreateOp.getLoc(), rewriter.getI4Type(), 0);
    auto zeroConstValue = zeroConstOp.getResult();

    SmallVector<SmallVector<Value>> array_values;
    SmallVector<Value> defVecHandles;

    auto arrayInputs = arrayCreateOp.getInputs();
    auto arrayLength = arrayInputs.size();
    auto elemWidth = hw::getBitWidth((arrayInputs[0].getType()));
    auto numChunks = (elemWidth + 3) / 4;
    for (size_t i = 0; i < static_cast<size_t>(numChunks); i++) {
      array_values.push_back({});
    }

    bool useMux = false;

    if (arrayInputs.size() <= SMALL_VEC_SIZE) {
      useMux = true;
      numSmallHWArrayInModule++;
    } else {
      for (const auto &arrayElem: arrayInputs) {
        // auto arrayElemWidth = hw::getBitWidth(arrayElem.getType());
        auto chunkValues = split_value_4B(arrayCreateOp.getOperation(), arrayElem, rewriter);
        for (size_t i = 0; i < chunkValues.size(); i++) {
          array_values[i].push_back(chunkValues[i]);
        }
      }

      for (auto elems: array_values) {
        auto newDefVecOp = rewriter.create<toucan::DefVectorOp>(arrayCreateOp.getLoc(), elems);
        defVecHandles.push_back(newDefVecOp.getHandle());
      }
    }

    for (auto userOp: arrayCreateOp->getUsers()) {
      if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(userOp)) {
        auto arrayIndex = extractMinimumWidth(arrayGetOp.getIndex(), rewriter, userOp);
        auto arrayIndexBits = hw::getBitWidth(arrayIndex.getType());

        auto maxIndexBits = llvm::Log2_64_Ceil(arrayLength);
        if (arrayIndexBits > maxIndexBits) {
          arrayGetOp.emitWarning() << "Too much index bits than necessary. (Expect index bits no more than " << maxIndexBits << ", but got " << arrayIndexBits;
        }

        if (useMux) {
          auto readResult = generate_mux_chain_for_array(arrayGetOp, rewriter, arrayInputs, arrayIndex);
          copyCustomizedAttrs(userOp, readResult.getDefiningOp());
          rewriter.replaceOp(userOp, readResult);
        } else {
          auto indicies_4b = split_value_4B(userOp, arrayIndex, rewriter);
          SmallVector<Value> arrayGetResults;

          for (auto vecHandle: defVecHandles) {
            auto readOp = rewriter.create<toucan::VectorReadOp>(arrayGetOp.getLoc(), vecHandle, zeroConstValue, indicies_4b);
            arrayGetResults.push_back(readOp.getResult());
          }

          if (arrayGetResults.size() > 1) {
            auto concatOp = rewriter.create<comb::ConcatOp>(arrayGetOp.getLoc(), arrayGetResults);

            copyCustomizedAttrs(userOp, concatOp);
            rewriter.replaceOp(arrayGetOp, concatOp);
          } else {
            auto result = arrayGetResults[0];
            copyCustomizedAttrs(userOp, result.getDefiningOp());
            rewriter.replaceOp(arrayGetOp, result);
          }
        }
      } else {
        userOp->emitError() << "Unknow op using const array";
        return failure();
      }
    }
    
    rewriter.eraseOp(arrayCreateOp);

    return success();
  }
};




struct LowerHWVectorTo4BPass : toucan::impl::LowerHWVectorTo4BBase<LowerHWVectorTo4BPass> {
  using LowerHWVectorTo4BBase<LowerHWVectorTo4BPass>::LowerHWVectorTo4BBase;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  std::shared_ptr<ConversionTarget> target;

  LogicalResult initialize(MLIRContext *context) override {
    numConstArrayInModule = 0;
    numHWArrayInModule = 0;
    numSmallHWArrayInModule = 0;
    numSmallConstArray = 0;

    RewritePatternSet owningPatterns(context);
    
    owningPatterns.add<LowerHWArrayTo4B>(context);
    owningPatterns.add<LowerConstArrayTo4B>(context);

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));

    ConversionTarget conversionTarget(*context);

    conversionTarget.addLegalDialect<toucan::ToucanDialect>();
    conversionTarget.addLegalDialect<hw::HWDialect>();
    conversionTarget.addLegalDialect<comb::CombDialect>();

    // After lowering, following ops should no longer appear
    conversionTarget.addIllegalOp<hw::ArrayCreateOp>();
    conversionTarget.addIllegalOp<hw::AggregateConstantOp>();
    target = std::make_shared<ConversionTarget>(std::move(
    conversionTarget));

    return success();
  }


  LogicalResult runOnModule(hw::HWModuleOp mod) {
    return applyFullConversion(mod, *target, *patterns);
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<hw::HWModuleOp> modulesToProcess;
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }
    // // Sequential
    // for (auto mod: modulesToProcess) {
    //   auto ret = runOnModule(mod);
    //   if (failed(ret)) return signalPassFailure();
    // }

    // Parallel
    auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
      return runOnModule(mod);
    });
    if (failed(result)) return signalPassFailure();

    numConstArray = numConstArrayInModule;
    numHWArray = numHWArrayInModule;
    numSmallHWArray = numSmallHWArrayInModule;
    numSmallConstArray = numSmallConstArrayInModule;
  }

};

std::unique_ptr<mlir::Pass> toucan::createLowerHWVectorTo4BPass() {
  return std::make_unique<LowerHWVectorTo4BPass>();
}
