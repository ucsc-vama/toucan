
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
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "toucan/ToucanTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>


#define GEN_PASS_DEF_LOWERSEQTO4B
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerSeqTo4BPass"

struct LowerSeqTo4BPass : toucan::impl::LowerSeqTo4BBase<LowerSeqTo4BPass> {
  using LowerSeqTo4BBase<LowerSeqTo4BPass>::LowerSeqTo4BBase;

  

  LogicalResult runOnModule(hw::HWModuleOp mod) {
    SmallVector<Operation*> toRemove;
    circt::DenseSet<mlir::Value> processedRegs;

    for (auto &stmt: mod.getOps()) {
      if (auto regOp = dyn_cast<toucan::DefRegOp>(stmt)) {
        auto regName = regOp.getSymName();
        auto regHandle = regOp.getHandle();
        auto regBitWidth = regHandle.getType().getElementWidth();

        

        OpBuilder builder(regOp);
        IRRewriter rewriter(builder);

        auto namehint = rewriter.getStringAttr(regName);

        // First, expand
        rewriter.setInsertionPointAfter(regOp);

        if (regBitWidth > 4) {
          SmallVector<std::tuple<IntegerAttr, mlir::Value, mlir::StringAttr>> newRegInfos;

          auto chunks = split_signal_4B(regBitWidth);
          for (auto &chunk: chunks) {
            auto regId = std::get<0>(chunk);
            auto regWidth = std::get<1>(chunk);

            auto newRegName = rewriter.getStringAttr(regName + "_Fragment_" + std::to_string(regId));
            auto regDataType = rewriter.getIntegerType(regWidth);

            auto newRegOp_4B = rewriter.create<toucan::DefRegOp>(regOp.getLoc(), newRegName, regDataType);

            auto newRegHandle = newRegOp_4B.getHandle();

            // set name hint
            setSVNameHintAttr(newRegOp_4B, namehint);
            // Set fragment Id
            auto fragmentId = rewriter.getIntegerAttr(rewriter.getIntegerType(32), regId);
            setSignalFragmentIDAttr(newRegOp_4B, fragmentId);

            newRegInfos.push_back({fragmentId, newRegHandle, namehint});
          } 
          toRemove.push_back(regOp);

          for (auto op: regHandle.getUsers()) {
            rewriter.setInsertionPointAfter(op);

            if (auto regReadOp = dyn_cast<toucan::RegReadOp>(op)) {
              // reads this register
              SmallVector<mlir::Value> regReadValues_4B;

              for (auto &each4BReg: newRegInfos) {
                auto regId_4b = std::get<0>(each4BReg);
                auto regHandle_4b = std::get<1>(each4BReg);
                auto regNameHint_4b = std::get<2>(each4BReg);

                auto regReadOp_4b = rewriter.create<toucan::RegReadOp>(regReadOp->getLoc(), regHandle_4b);

                setSVNameHintAttr(regReadOp_4b, regNameHint_4b);
                setSignalFragmentIDAttr(regReadOp_4b, regId_4b);

                regReadValues_4B.push_back(regReadOp_4b.getResult());
              }

              auto bitAggregator = rewriter.create<toucan::BitAggregateOp>(regReadOp->getLoc(), regReadValues_4B);
              rewriter.replaceAllUsesWith(regReadOp, bitAggregator);

            } else if (auto regWriteOp = dyn_cast<toucan::RegWriteOp>(op)) {
              // writes to this register
              auto writeData = regWriteOp.getData();
              auto bitScatter = rewriter.create<toucan::BitScatterOp>(regWriteOp->getLoc(), writeData);

              auto signalsToWrite = bitScatter.getOutputs();

              assert(newRegInfos.size() == signalsToWrite.size());

              for (auto &each4BReg: newRegInfos) {
                auto regId_4b = std::get<0>(each4BReg);
                auto regHandle_4b = std::get<1>(each4BReg);
                // auto regNameHint_4b = std::get<2>(each4BReg);

                auto writeData_4b = signalsToWrite[regId_4b.getInt()];

                rewriter.create<toucan::RegWriteOp>(regWriteOp->getLoc(), writeData_4b, regHandle_4b);

                // setSVNameHintAttr(regReadOp_4b, regNameHint_4b);
                // setSignalFragmentIDAttr(regReadOp_4b, regId_4b);
              }

            } else {
              op->emitError("The register can only be accessed by either a RegReadOp or RegWriteOp");
              return failure();
            }
            toRemove.push_back(op);
          }

        } else {
          // Don't need expand
          // set name hint
          setSVNameHintAttr(regOp, namehint);
          // Set fragment Id
          auto fragmentId = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
          setSignalFragmentIDAttr(regOp, fragmentId);

        }
      } else if (auto memOp = dyn_cast<seq::FirMemOp>(stmt)) {
        // Memory
        auto memValue = memOp.getMemory();
        auto memName = memOp.getName().value_or(StringRef(""));
        auto firMemDataType = memValue.getType();
        auto memDepth = firMemDataType.getDepth();
        auto memWidth = firMemDataType.getWidth();
        auto maskLaneWidth = 0;


        if (auto maskWidth = firMemDataType.getMaskWidth()) {
          maskLaneWidth = memWidth / (*maskWidth);
          if (memWidth % (*maskWidth) != 0) {
            return memOp.emitError() << "Incorrect mask width, got " << (*maskWidth) << " for a " << memWidth << " width memory";
          }
        }

        // TODO: 

        // if (memWidth > 4) {
        //   SmallVector<std::tuple<IntegerAttr, mlir::Value, mlir::StringAttr>> newMemInfos;

        //   auto chunks = split_signal_4B(memWidth);
        //   for (auto &chunk: chunks) {
        //     // Split to 4-bit wide memory

        //   }
        // }


      }
    }
  



    // for (auto op: llvm::reverse(toRemove)) op->erase();

    return success();
  }

  void runOnOperation() final {
    auto mod = getOperation();

    SmallVector<hw::HWModuleOp> modulesToProcess;
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        modulesToProcess.push_back(mod);
      }
    }
    // Sequential
    for (auto mod: modulesToProcess) {
      auto ret = runOnModule(mod);
      if (failed(ret)) return signalPassFailure();
    }

    // Parallel
    // auto result = mlir::failableParallelForEach(&getContext(), modulesToProcess.begin(), modulesToProcess.end(), [&](auto mod) {
    //   return runOnModule(mod);
    // });
    // if (failed(result)) return signalPassFailure();
  }

};

std::unique_ptr<mlir::Pass> toucan::createLowerSeqTo4BPass() {
  return std::make_unique<LowerSeqTo4BPass>();
}
