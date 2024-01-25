
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


#define GEN_PASS_DEF_LOWERREGMEMTO4B
#include "toucan/ToucanPassCommon.h"

#include "toucan/ToucanOps.h"
#include "toucan/ToucanUtils.h"

using namespace toucan;
using namespace circt;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "LowerRegMemTo4BPass"

struct LowerRegMemTo4BPass : toucan::impl::LowerRegMemTo4BBase<LowerRegMemTo4BPass> {
  using LowerRegMemTo4BBase<LowerRegMemTo4BPass>::LowerRegMemTo4BBase;

  

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
          SmallVector<std::tuple<IntegerAttr, int, mlir::Value, mlir::StringAttr>> newRegInfos;

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

            newRegInfos.push_back({fragmentId, regWidth, newRegHandle, namehint});
          } 
          toRemove.push_back(regOp);

          for (auto op: regHandle.getUsers()) {
            rewriter.setInsertionPointAfter(op);

            if (auto regReadOp = dyn_cast<toucan::RegReadOp>(op)) {
              // reads this register
              SmallVector<mlir::Value> regReadValues_4B;

              for (auto &each4BReg: newRegInfos) {
                auto regId_4b = std::get<0>(each4BReg);
                // auto regWidth_4b = std::get<1>(each4BReg);
                auto regHandle_4b = std::get<2>(each4BReg);
                auto regNameHint_4b = std::get<3>(each4BReg);

                auto regReadOp_4b = rewriter.create<toucan::RegReadOp>(regReadOp->getLoc(), regHandle_4b);

                setSVNameHintAttr(regReadOp_4b, regNameHint_4b);
                setSignalFragmentIDAttr(regReadOp_4b, regId_4b);

                regReadValues_4B.push_back(regReadOp_4b.getResult());
              }

              // auto bitAggregator = rewriter.create<toucan::BitAggregateOp>(regReadOp->getLoc(), regReadValues_4B);
              // rewriter.replaceAllUsesWith(regReadOp, bitAggregator);
              auto bitConcatOp = rewriter.create<comb::ConcatOp>(regReadOp->getLoc(), regReadValues_4B);
              rewriter.replaceAllUsesWith(regReadOp, bitConcatOp);

            } else if (auto regWriteOp = dyn_cast<toucan::RegWriteOp>(op)) {
              // writes to this register
              auto writeData = regWriteOp.getData();

              for (auto &each4BReg: newRegInfos) {
                auto regId_4b = std::get<0>(each4BReg);
                auto regWidth_4b = std::get<1>(each4BReg);
                auto regHandle_4b = std::get<2>(each4BReg);
                // auto regNameHint_4b = std::get<3>(each4BReg);

                auto signalExtractOp = rewriter.create<comb::ExtractOp>(regWriteOp.getLoc(), writeData, regId_4b.getInt() * 4, regWidth_4b);
                auto writeData_4b = signalExtractOp.getResult();

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
      } else if (auto memOp = dyn_cast<toucan::DefMemOp>(stmt)) {
        // Memory
        auto memValue = memOp.getHandle();
        auto memName = memOp.getSymName();
        auto memType = memValue.getType();
        auto memDepth = memType.getDepth();
        auto memWidth = memType.getElementWidth();

        OpBuilder builder(memOp);
        IRRewriter rewriter(builder);

        auto namehint = rewriter.getStringAttr(memName);

        if (memWidth > 4) {
          SmallVector<std::tuple<IntegerAttr, int, mlir::Value, mlir::StringAttr>> newMemInfos;

          auto chunks = split_signal_4B(memWidth);
          for (auto &chunk: chunks) {
            auto newMemId = std::get<0>(chunk);
            auto newMemWidth = std::get<1>(chunk);

            auto newMemName = rewriter.getStringAttr(memName + "_Fragment_" + std::to_string(newMemId));
            auto newMemElemType = rewriter.getIntegerType(newMemWidth);
            auto newMemDataType = rewriter.getType<toucan::MemType>(memDepth, newMemElemType);

            auto newMemOp_4B = rewriter.create<toucan::DefMemOp>(memOp.getLoc(), newMemDataType, newMemName);

            auto newMemHandle = newMemOp_4B.getHandle();

            // set name hint
            setSVNameHintAttr(newMemOp_4B, namehint);
            // Set fragment Id
            auto fragmentId = rewriter.getIntegerAttr(rewriter.getIntegerType(32), newMemId);
            setSignalFragmentIDAttr(newMemOp_4B, fragmentId);

            newMemInfos.push_back({fragmentId, newMemWidth, newMemHandle, namehint});
          } 
          toRemove.push_back(memOp);


          for (auto op: memOp->getUsers()) {
            rewriter.setInsertionPointAfter(op);

            if (auto memReadOp = dyn_cast<toucan::MemReadOp>(op)) {
              //
              SmallVector<mlir::Value> memReadValues_4B;
              auto memReadEn = memReadOp.getEn();
              auto memReadAddr = memReadOp.getAddr();

              for (auto &each4BMem: newMemInfos) {
                auto memId_4b = std::get<0>(each4BMem);
                auto memHandle_4b = std::get<2>(each4BMem);
                auto memNameHint_4b = std::get<3>(each4BMem);

                auto memReadOp_4b = rewriter.create<toucan::MemReadOp>(memReadOp->getLoc(), memHandle_4b, memReadAddr, memReadEn);

                setSVNameHintAttr(memReadOp_4b, memNameHint_4b);
                setSignalFragmentIDAttr(memReadOp_4b, memId_4b);

                memReadValues_4B.push_back(memReadOp_4b.getResult());
              }

              // auto bitAggregator = rewriter.create<toucan::BitAggregateOp>(regReadOp->getLoc(), regReadValues_4B);
              // rewriter.replaceAllUsesWith(regReadOp, bitAggregator);
              auto bitConcatOp = rewriter.create<comb::ConcatOp>(memReadOp->getLoc(), memReadValues_4B);
              rewriter.replaceAllUsesWith(memReadOp, bitConcatOp);

            } else if (auto memWriteOp = dyn_cast<toucan::MemWriteOp>(op)) {
              // todo
              auto memWriteData = memWriteOp.getData();
              auto memWriteAddr = memWriteOp.getAddr();
              auto memWriteEn = memWriteOp.getEn();

              for (auto &each4BMem: newMemInfos) {
                auto memId_4b = std::get<0>(each4BMem);
                auto memWidth_4b = std::get<1>(each4BMem);
                auto memHandle_4b = std::get<2>(each4BMem);
                // auto memNameHint_4b = std::get<3>(each4BMem);

                auto signalExtractOp = rewriter.create<comb::ExtractOp>(memWriteOp.getLoc(), memWriteData, memId_4b.getInt() * 4, memWidth_4b);
                auto writeData_4b = signalExtractOp.getResult();

                rewriter.create<toucan::MemWriteOp>(memWriteOp->getLoc(), memHandle_4b, memWriteAddr, writeData_4b, memWriteEn);

                // setSVNameHintAttr(regReadOp_4b, regNameHint_4b);
                // setSignalFragmentIDAttr(regReadOp_4b, regId_4b);
              }
            } else {
              op->emitError("The memory can only be accessed by either a MemReadOp or MemWriteOp");
              return failure();
            }
            toRemove.push_back(op);
          }
        } else {
          // memory width less or equal than 4
          // Don't need expand
          // set name hint
          setSVNameHintAttr(memOp, namehint);
          // Set fragment Id
          auto fragmentId = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
          setSignalFragmentIDAttr(memOp, fragmentId);
        }

      }
    }

    for (auto op: llvm::reverse(toRemove)) op->erase();

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
  }

};

std::unique_ptr<mlir::Pass> toucan::createLowerRegMemTo4BPass() {
  return std::make_unique<LowerRegMemTo4BPass>();
}
